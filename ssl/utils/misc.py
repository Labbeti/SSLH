
import os.path as osp

from argparse import Namespace

from mlu.optim import CosineLRScheduler, SoftCosineLRScheduler

from ssl.datasets.abc import DatasetInterface
from ssl.models.checkpoint import CheckPointABC, CheckPointMultiple
from ssl.utils.radam import RAdam, PlainRAdam, AdamW

from torch import Tensor
from torch.nn import Module
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Any, List, Optional, TypeVar, Union


def build_optimizer(args: Namespace, model: Module) -> Optimizer:
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
		raise RuntimeError("Unknown optimizer \"{:s}\".".format(str(args.optim)))

	return optim


def build_scheduler(args: Namespace, optim: Optimizer) -> Optional[LambdaLR]:
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
		scheduler = CosineLRScheduler(optim, nb_epochs=args.nb_epochs)
	elif name in ["MultiStepLR".lower(), "step"]:
		scheduler = MultiStepLR(optim, milestones=args.epoch_steps, gamma=args.lr_decay_ratio)
	elif name in ["SoftCosineLRScheduler".lower(), "softcos"]:
		scheduler = SoftCosineLRScheduler(optim, nb_epochs=args.nb_epochs)
	else:
		scheduler = None

	return scheduler


def get_prefix(
	args: Namespace, fold_val: Optional[List[int]], interface: DatasetInterface, start_date: str, model_name: str, train_name: str
) -> str:
	"""
		Returns the default prefix used for tensorboard directory.

		:param args: The argparse arguments. Must contains the attributes "supervised_ratio" and "tag".
		:param fold_val: The validations folds used. Can be None if no cross-validation are currently activated.
		:param interface: The DatasetInterface used for training the model.
		:param start_date: The date of the start of the program get by the "get_datetime()" function.
		:param model_name: The class name of the model to train.
		:param train_name: The name of the method used to train the model (ex: MixMatch, FixMatch...).
		:returns: The prefix to use for tensorboard folder name.
	"""
	fold_suffix = "" if fold_val is None else str(fold_val).replace(" ", "")
	su_ratio = args.supervised_ratio * 100.0
	prefix = "{:s}_{:s}_{:s}_{:s}_{:.2f}%_{:s}_{:s}".format(
		interface.get_dataset_name(), start_date, model_name, train_name, su_ratio, fold_suffix, args.tag)
	return prefix


def build_tensorboard_writer(args: Namespace, prefix: str) -> (Optional[SummaryWriter], str):
	"""
		Build SummaryWriter for saving data in tensorboard files.

		:param args: The argparse arguments. Must contains the string attribute "logdir" and the boolean attribute "write_results".
		:param prefix: The prefix used for the tensorboard writer directory name.
		:returns: The tensorboard SummaryWriter object for saving results and the path to the tensorboard directory.
	"""
	dirpath_writer = osp.join(args.logdir, "{:s}".format(prefix))
	writer = SummaryWriter(dirpath_writer) if args.write_results else None
	return writer, dirpath_writer


def build_checkpoint(
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


T = TypeVar("T")


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
