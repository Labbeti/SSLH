
import json
import os.path as osp
import torch
import torchaudio

from argparse import Namespace

from mlu.metrics import (
	Metric, MetricWrapper, CategoricalAccuracy, FScore, AveragePrecision, RocAuc, DPrime
)
from mlu.nn import CrossEntropyWithVectors, Max
from mlu.optim import CosineLRScheduler, SoftCosineLRScheduler
from mlu.utils.misc import reset_seed, get_datetime, get_current_git_hash, get_nb_parameters

from sslh.datasets.base import DatasetBuilder
from sslh.datasets.dataset_sized import DatasetSized
from sslh.datasets.get_builder import get_dataset_builder
from sslh.models.checkpoint import CheckPointABC, CheckPointMultiple
from sslh.utils.cross_validation import cross_validation
from sslh.utils.radam import RAdam, PlainRAdam, AdamW
from sslh.utils.recorder import RecorderABC
from sslh.validation.validater_stack import ValidaterStack

from time import time
from torch import Tensor
from torch.nn import Module, BCELoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Any, Callable, Dict, List, Optional, Sized, Tuple, TypeVar, Union


T = TypeVar("T")


def get_optimizer(args: Namespace, model: Module) -> Optimizer:
	"""
		Instantiate optimizer from args and torch module.
		Available optimizers : Adam, SGD, RAdam, PlainRAdam and AdamW.

		:param args: The argparse arguments. Must contains the attributes "optimizer", learning_rate and weight_decay.
			If args.scheduler == "SGD", must contains the attribute "momentum" and "use_nesterov".
		:param model: The torch module to update with the optimizer.
		:returns: The optimizer to use.
	"""
	name = args.optimizer.lower()

	kwargs = dict(params=model.parameters(), lr=args.learning_rate)
	if args.weight_decay is not None:
		kwargs["weight_decay"] = args.weight_decay

	if name == "Adam".lower():
		optim = Adam(**kwargs)

	elif name == "SGD".lower():
		if args.momentum is not None:
			kwargs["momentum"] = args.momentum
		if args.use_nesterov is not None:
			kwargs["nesterov"] = args.use_nesterov
		optim = SGD(**kwargs)

	elif name == "RAdam".lower():
		optim = RAdam(**kwargs)

	elif name == "PlainRAdam".lower():
		optim = PlainRAdam(**kwargs)

	elif name == "AdamW".lower():
		optim = AdamW(**kwargs)

	else:
		raise RuntimeError("Unknown optimizer '{:s}'.".format(str(args.optim)))

	return optim


def get_scheduler(args: Namespace, optim: Optimizer) -> Optional[LambdaLR]:
	"""
		Instantiate scheduler from args and optimizer.
		Available schedulers : CosineLRScheduler, MultiStepLR and SoftCosineLRScheduler.

		:param args: The argparse arguments. Must contains the attribute "scheduler".
			If args.scheduler == "CosineLRScheduler", must contains the attribute "nb_epochs".
			If args.scheduler == "MultiStepLR", must contains the attributes "epoch_steps" and "gamma".
			If args.scheduler == "SoftCosineLRScheduler", must contains the attribute "nb_epochs".
		:param optim: The torch Optimizer to update with the scheduler.
		:returns: The scheduler to use. If "args.scheduler" is not an available scheduler then returns None.
	"""
	name = str(args.scheduler).lower()

	if name in ["CosineLRScheduler".lower(), "cosine"]:
		nb_steps = args.sched_nb_steps if args.sched_nb_steps is not None else args.nb_epochs
		scheduler = CosineLRScheduler(optim, nb_steps=nb_steps, coefficient=args.sched_coef)
	elif name in ["MultiStepLR".lower(), "step"]:
		scheduler = MultiStepLR(optim, milestones=args.epoch_steps, gamma=args.lr_decay_ratio)
	elif name in ["SoftCosineLRScheduler".lower(), "softcos"]:
		scheduler = SoftCosineLRScheduler(optim, nb_steps=args.nb_epochs)
	else:
		scheduler = None

	return scheduler


def get_prefix(
	args: Namespace,
	fold_val: Optional[List[int]],
	builder: DatasetBuilder,
	start_date: str,
	model_name: str,
	run_name: str,
) -> str:
	"""
		Returns the default prefix used for tensorboard directory.

		:param args: The argparse arguments. Must contains the attributes "supervised_ratio" and "tag".
		:param fold_val: The validations folds used. Can be None if no cross-validation are currently activated.
		:param builder: The DatasetBuilder used for training the model.
		:param start_date: The date of the start of the program get by the "get_datetime()" function.
		:param model_name: The class name of the model to train.
		:param run_name: The name of the method used to train the model (ex: MixMatch, FixMatch...).
		:returns: The prefix to use for tensorboard folder name.
	"""
	fold_suffix = "" if fold_val is None else str(fold_val).replace(" ", "")
	su_ratio = args.supervised_ratio * 100.0
	prefix = "{:s}_{:s}_{:s}_{:s}_{:.2f}%_{:s}_{:s}".format(
		builder.get_dataset_name(), start_date, model_name, run_name, su_ratio, fold_suffix, args.tag)
	return prefix


def get_tensorboard_writer(args: Namespace, prefix: str) -> (Optional[SummaryWriter], str):
	"""
		Build SummaryWriter for saving data in tensorboard files.

		:param args: The argparse arguments. Must contains the string attribute "logdir" and the boolean attribute "write_results".
		:param prefix: The prefix used for the tensorboard writer directory name.
		:returns: The tensorboard SummaryWriter object for saving results and the path to the tensorboard directory.
	"""
	dirpath_writer = osp.join(args.logdir, prefix)
	writer = SummaryWriter(dirpath_writer) if args.write_results else None
	return writer, dirpath_writer


def get_checkpoint(
	args: Namespace, dirpath_writer: str, model: Module, optim: Optimizer
) -> Optional[CheckPointABC]:
	"""
		Build a checkpoint object for saving best model and optimizer.

		:param args: The argparse arguments. Must contains the boolean attribute "write_results".
		:param dirpath_writer: The path to the directory where the model files will be saved.
		:param model: The torch Module to save in file.
		:param optim: The torch Optimizer to save in file.
		:returns: The checkpoint class if "args.write_results" is True, otherwise returns None.
	"""
	checkpoint = CheckPointMultiple(
		model, optim, dirpath_writer, model.__class__.__name__, save_in_file=args.write_results)
	return checkpoint


def get_activation(name: str, clamp: bool = True, clamp_min: float = 2e-30) -> Callable:
	"""
		Build an activation function.

		:param name: The name of the activation function. Can be "softmax" or "sigmoid".
		:param clamp: If True, add a clamp operation after the activation.
		:param clamp_min: The minimal value of the clamp. Unused if clamp == False.
		:return: The activation function.
	"""
	name = str(name).lower()

	if name == "softmax":
		if clamp:
			activation = lambda x, dim: x.softmax(dim=dim).clamp(min=clamp_min)
		else:
			activation = torch.softmax

	elif name == "sigmoid":
		if clamp:
			activation = lambda x, dim: x.sigmoid().clamp(min=clamp_min)
		else:
			activation = lambda x, dim: torch.sigmoid(x)

	elif name == "log_softmax":
		if clamp:
			activation = lambda x, dim: x.log_softmax(dim=dim).clamp(min=clamp_min)
		else:
			activation = torch.log_softmax

	elif name in ["none", "identity"]:
		if clamp:
			activation = lambda x, dim: x.clamp(min=clamp_min)
		else:
			activation = lambda x, dim: x

	else:
		raise RuntimeError("Unknown activation function '{name}'.")

	return activation


def to_dict_rec(obj: Any, class_name_key: Optional[str] = "__class__") -> Union[dict, list]:
	"""
		Convert an object to a dictionary.

		Source code was imported from : (with few changes)
			https://stackoverflow.com/questions/1036409/recursively-convert-python-object-graph-to-dictionary

		:param obj: The object to convert.
		:param class_name_key: Key used to save the class name if we convert an object.
		:returns: The dictionary corresponding to the object.
	"""
	if isinstance(obj, dict):
		return {
			key: to_dict_rec(value, class_name_key)
			for key, value in obj.items()
		}
	elif isinstance(obj, Tensor):
		return to_dict_rec(obj.tolist(), class_name_key)
	elif hasattr(obj, "_ast"):
		return to_dict_rec(obj._ast())
	elif hasattr(obj, "__iter__") and not isinstance(obj, str):
		return [to_dict_rec(v, class_name_key) for v in obj]
	elif hasattr(obj, "__dict__"):
		data = {}
		if class_name_key is not None and hasattr(obj, "__class__"):
			data[class_name_key] = obj.__class__.__name__
		data.update(dict([
			(attr, to_dict_rec(value, class_name_key))
			for attr, value in obj.__dict__.items()
			if not callable(value) and not attr.startswith('_')
		]))
		return data
	else:
		return obj


def interpolation(min_: float, max_: float, coefficient: float) -> float:
	"""
		Compute the linear interpolation between min_ and max_ with a coefficient.

		:param min_: The minimal value used for interpolation.
		:param max_: The maximal value used for interpolation.
		:param coefficient: The coefficient in [0.0, 1.0] for compute the results.
		:returns: The value interpolated between min_ and max_.
	"""
	return (max_ - min_) * coefficient + min_


def normalize(value: T, old_min: T, old_max: T, new_min: T = 0.0, new_max: T = 1.0) -> T:
	"""
		Normalize a value from range [old_min, old_max] to [new_min, new_max].

		:param value: The value to normalize.
		:param old_min: The minimal value of the previous range.
		:param old_max: The maximal value of the previous range.
		:param new_min: The minimal value of the new range. (default: 0.0)
		:param new_max: The maximal value of the new range. (default: 1.0)
		:returns: The value normalized in the new range.
	"""
	return (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def guess_folds(
	folds_train: Optional[List[int]], folds_val: Optional[List[int]], folds: Optional[List[int]]
) -> (Optional[List[int]], Optional[List[int]]):
	"""
		Use the folds_train and folds to guess folds_val OR use the folds_val and folds to guess folds_train.

		:param folds_train: The training folds.
		:param folds_val: The validation folds.
		:param folds: The list of folds.
		:return: A tuple of folds (training folds, validation folds).
	"""
	if folds is not None:
		if folds_train is None and folds_val is not None:
			folds_train = list(set(folds).difference(folds_val))
		elif folds_train is not None and folds_val is None:
			folds_val = list(set(folds).difference(folds_train))
		elif folds_train is None and folds_val is None:
			folds_val = [len(folds)]
			folds_train = list(set(folds).difference(folds_val))
	else:
		folds_train, folds_val = None, None

	return folds_train, folds_val


def main_run(create_args_fn: Callable, run_fn: Callable, run_name: str):
	# Initialisation
	start_time = time()
	start_date = get_datetime()
	git_hash = get_current_git_hash()

	args = create_args_fn()
	reset_seed(args.seed)
	torch.autograd.set_detect_anomaly(args.debug)
	torchaudio.set_audio_backend("sox_io")

	if args.debug:
		print(json.dumps(args.__dict__, indent="\t"))

	if "cuda" in args.device:
		if not torch.cuda.is_available():
			raise RuntimeError(f"Cannot use device '{args.device}' because CUDA is not available.")
		torch.cuda.empty_cache()

	device = torch.device(args.device)

	print(f"Start {run_name}. (tag: '{args.tag}')")
	print(f" - start_date: {start_date}")
	print(f" - su_ratio: {args.supervised_ratio}")

	builder = get_dataset_builder(args.dataset_name)

	# Run
	if args.cross_validation:
		cross_validation(run_fn, args, start_date, git_hash, builder, run_name, device)
	else:
		folds_train, folds_val = guess_folds(args.folds_train, args.folds_val, builder.get_folds())
		bests = run_fn(args, start_date, git_hash, builder, folds_train, folds_val, device)

		# Print min and max for all metrics recorded
		print_best_scores(bests)

	exec_time = "{:.2f}".format(time() - start_time)
	print()
	print(f"Program started at '{start_date}' and terminated at '{get_datetime()}'.")
	print(f"Total execution time: {exec_time}s.")


def print_best_scores(bests: Dict[str, Dict[str, float]]):
	name_size = max([len(metric_name) for metric_name in bests.keys()])
	name_format = f"{{:{name_size}s}}"
	name_format_title = name_format

	float_size = 10
	float_format = "{:.4e}"
	float_format_title = f"{{:{float_size}s}}"

	int_size = max([max(len(str(min_max["idx_min"])), len(str(min_max["idx_max"]))) for min_max in bests.values()])
	int_size = max(int_size, len("epoch"))
	int_format = f"{{:{int_size}d}}"
	int_format_title = f"{{:{int_size}s}}"

	title_format = " | ".join(
		(name_format_title, float_format_title, int_format_title, float_format_title, int_format_title))
	line = title_format.format("Metric", "max", "epoch", "min", "epoch")
	print(f"| {line} |")
	line = title_format.format("-" * name_size, "-" * float_size, "-" * int_size, "-" * float_size, "-" * int_size)
	print(f"| {line} |")

	for metric_name, min_max in bests.items():
		format_ = " | ".join((name_format, float_format, int_format, float_format, int_format))
		line = format_.format(metric_name, min_max["max"], min_max["idx_max"], min_max["min"], min_max["idx_min"])
		print(f"| {line} |")


def print_start_run_info(
	args: Namespace,
	dataset_train: Sized,
	dataset_val: Sized,
	dataset_eval: Optional[Sized],
	folds_train: Optional[List[int]],
	folds_val: Optional[List[int]],
	model: Module,
	run_name: str,
) -> float:
	print("Dataset : {:s} (train={:d}, val={:d}, eval={:s}, folds_train={:s}, folds_val={:s}).".format(
		args.dataset_name,
		len(dataset_train),
		len(dataset_val),
		str(len(dataset_eval)) if dataset_eval is not None else "None",
		str(folds_train),
		str(folds_val)
	))
	print("Model: {:s} ({:d} parameters).".format(model.__class__.__name__, get_nb_parameters(model)))
	print()
	print("Start {:s} training with {:d} epochs (tag: '{:s}')...".format(run_name, args.nb_epochs, args.tag))
	start_time = time()
	return start_time


def print_end_run_info(run_name: str, start_time: float) -> float:
	duration = time() - start_time
	print()
	print("End {:s} training. (duration = {:.2f})".format(run_name, duration))
	return duration


def get_default_metrics(
	target_type: str,
) -> Tuple[str, Dict[str, Metric], Dict[str, Metric], Dict[str, Metric]]:
	if target_type == "monolabel":
		main_metric_name = "val/acc"

		metrics_train = {"train/acc": CategoricalAccuracy(dim=1)}
		metrics_val = {
			"val/acc": CategoricalAccuracy(dim=1),
			"val/ce": MetricWrapper(CrossEntropyWithVectors(dim=1)),
			"val/max": MetricWrapper(Max(dim=1), use_target=False, reduce_fn=torch.mean),
		}
		metrics_eval = {name.replace("val", "eval"): metric for name, metric in metrics_val.items()}

	elif target_type == "multilabel":
		main_metric_name = "val/fscore"

		metrics_train = {"train/fscore": FScore()}
		metrics_val = {"val/fscore": FScore()}
		metrics_eval = {}

	else:
		raise RuntimeError(f"Unknown target type '{target_type}'.")

	return main_metric_name, metrics_train, metrics_val, metrics_eval


def evaluate(
	args: Namespace,
	model: Module,
	activation: Callable,
	builder: DatasetBuilder,
	checkpoint: Optional[CheckPointABC],
	recorder: RecorderABC,
	dataset_val: DatasetSized,
	dataset_eval: DatasetSized,
):
	target_type = builder.get_target_type()
	if target_type == "monolabel":
		metrics_val_stack = {}
	elif target_type == "multilabel":
		metrics_val_stack = {
			"val_stack/bce": MetricWrapper(BCELoss()),
			"val_stack/fscore": FScore(),
			"val_stack/mAP": AveragePrecision(),
			"val_stack/mAUC": RocAuc(),
			"val_stack/dPrime": DPrime(),
		}
	else:
		raise RuntimeError(f"Unknown target type '{target_type}'.")

	metrics_eval_stack = {name.replace("val_stack", "eval_stack"): metric for name, metric in metrics_val_stack.items()}

	if checkpoint is not None and checkpoint.is_saved():
		print("Loading best model for evaluation...")
		checkpoint.load_best_state(model=model, optim=None)
	else:
		print("No model saved for evaluation, use last model.")

	recorder.set_storage(write_mean=True, write_std=False, write_min_mean=False, write_max_mean=False)

	if len(metrics_val_stack) > 0:
		loader_val = builder.get_loader_val(dataset_val, args.batch_size_s, drop_last=False, num_workers=0)
		validater_stack = ValidaterStack(model, activation, loader_val, metrics_val_stack, recorder, name="val_stack")
		validater_stack.add_callback_on_end(recorder)
		validater_stack.val(0)
		print()

	if len(metrics_eval_stack) > 0 and builder.has_evaluation():
		loader_eval = builder.get_loader_val(dataset_eval, args.batch_size_s, drop_last=False, num_workers=0)
		evaluator_stack = ValidaterStack(model, activation, loader_eval, metrics_eval_stack, recorder, name="eval_stack")
		evaluator_stack.add_callback_on_end(recorder)
		evaluator_stack.val(0)
		print()

	print()
