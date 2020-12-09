"""
	Main script for running a MixMatch training.
"""

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import torch

from argparse import ArgumentParser, Namespace

from sslh.datasets.get_interface import get_dataset_interface, DatasetInterface
from sslh.datasets.wrappers.multiple_dataset import MultipleDataset
from sslh.datasets.wrappers.no_label_dataset import NoLabelDataset
from sslh.mixmatch.loss import MixMatchLoss
from sslh.mixmatch.trainer import MixMatchTrainer
from sslh.utils.args import post_process_args, check_args, add_common_args
from sslh.utils.cross_validation import cross_validation
from sslh.utils.misc import build_optimizer, get_datetime, reset_seed, build_tensorboard_writer, build_scheduler, get_prefix, build_checkpoint
from sslh.utils.other_metrics import CategoricalAccuracyOnehot, CrossEntropyMetric, EntropyMetric, MaxMetric
from sslh.utils.recorder.recorder import Recorder
from sslh.utils.save import save_results
from sslh.utils.torch import CrossEntropyWithVectors
from sslh.utils.types import str_to_optional_str
from sslh.utils.zip_cycle import ZipCycle
from sslh.validation.validater import Validater

from time import time
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Subset
from typing import Optional, Dict, Union

TRAIN_NAME = "MixMatch"


def create_args() -> Namespace:
	parser = ArgumentParser()
	add_common_args(parser)

	group_mm = parser.add_argument_group(TRAIN_NAME)
	group_mm.add_argument("--lambda_u", type=float, default=1.0,
		help="MixMatch, FixMatch and ReMixMatch \"lambda_u\" hyperparameter. "
			 "Coefficient of unsupervised loss component. (default: 1.0)")
	group_mm.add_argument("--batch_size_u", "--bsize_u", type=int, default=128,
		help="Batch size used for unsupervised loader. (default: 128)")

	group_mm.add_argument("--nb_augms", type=int, default=2,
		help="Nb of augmentations used in MixMatch. (default: 2)")
	group_mm.add_argument("--temperature", "--sharpen_temperature", type=float, default=0.5,
		help="MixMatch and ReMixMatch hyperparameter temperature used by sharpen function. (default: 0.5)")
	group_mm.add_argument("--alpha", "--mixup_alpha", type=float, default=0.75,
		help="MixMatch and ReMixMatch hyperparameter \"alpha\" used by MixUp. (default: 0.75)")

	group_mm.add_argument("--criterion_u", type=str, default="ce",
		choices=["mse", "ce"],
		help="MixMatch unsupervised loss component. (default: ce)")

	group_mm.add_argument("--warmup_nb_steps", type=int, default=16000,
		help="Nb of steps when lambda_u and lambda_u1 is increase from 0 to their value. "
			 "Use 0 for deactivate warmup. (default: 16000)")

	group_mm.add_argument("--augm_weak", type=str_to_optional_str, default="weak",
		help="Augment pool for weak augmentation to use. (default: weak)")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	return args


def run_mixmatch(args: Namespace, start_date: str, fold_val: Optional[int], interface: DatasetInterface) -> Dict[str, Union[float, int]]:
	# Build loaders
	if interface.get_nb_folds() is None:
		folds_train, folds_val = None, None
	else:
		if fold_val is None:
			fold_val = interface.get_nb_folds()
		folds_train = list(range(1, interface.get_nb_folds() + 1))
		folds_train.remove(fold_val)
		folds_val = [fold_val]

	dataset_train_augm_weak = interface.get_dataset_train_augm_weak(args, folds_train)
	dataset_val = interface.get_dataset_val(args, folds_val)
	dataset_eval = interface.get_dataset_eval(args, None)

	indexes_s, indexes_u = interface.get_indexes(dataset_train_augm_weak, [args.supervised_ratio, 1.0 - args.supervised_ratio])
	dataset_train_s_augm_weak = Subset(dataset_train_augm_weak, indexes_s)
	dataset_train_u_augm_weak = Subset(dataset_train_augm_weak, indexes_u)
	dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)

	# Apply callables on the same batch u
	dataset_train_u_weak_multiple = MultipleDataset([dataset_train_u_augm_weak] * args.nb_augms)

	loader_train_s_augm = DataLoader(
		dataset=dataset_train_s_augm_weak, batch_size=args.batch_size_s, shuffle=True, num_workers=2, drop_last=True)
	loader_train_u_augms = DataLoader(
		dataset=dataset_train_u_weak_multiple, batch_size=args.batch_size_u, shuffle=True, num_workers=6, drop_last=True)

	loader_train = ZipCycle([loader_train_s_augm, loader_train_u_augms])
	loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=False)

	# Prepare model
	model = interface.build_model(args.model, args)
	model_name = model.__class__.__name__
	optim = build_optimizer(args, model)
	activation = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)

	criterion_s = CrossEntropyWithVectors()
	criterion_u = MSELoss() if args.criterion_u in ["mse"] else CrossEntropyWithVectors()
	criterion = MixMatchLoss(criterion_s, criterion_u)

	sched = build_scheduler(args, optim)

	# Prepare metrics
	main_metric_name = "val/acc"
	metrics_train_s_mix = {"acc_s_mix": CategoricalAccuracyOnehot(dim=1)}
	metrics_train_u_mix = {"acc_u_mix": CategoricalAccuracyOnehot(dim=1)}
	metrics_val = {
		"acc": CategoricalAccuracyOnehot(dim=1),
		"ce": CrossEntropyMetric(dim=1),
		"entropy": EntropyMetric(dim=1),
		"max": MaxMetric(dim=1),
	}

	# Prepare objects for saving data
	prefix = get_prefix(args, folds_val, interface, start_date, model_name, TRAIN_NAME)
	writer, dirpath_writer = build_tensorboard_writer(args, prefix)
	recorder = Recorder(writer)
	checkpoint = build_checkpoint(args, dirpath_writer, model, optim)

	# Start main training
	trainer = MixMatchTrainer(
		model,
		activation,
		optim,
		loader_train,
		metrics_train_s_mix,
		metrics_train_u_mix,
		recorder,
		criterion,
		temperature=args.temperature,
		alpha=args.alpha,
		lambda_u=args.lambda_u,
		warmup_nb_steps=args.warmup_nb_steps,
	)
	validator = Validater(model, activation, loader_val, metrics_val, recorder, checkpoint=checkpoint, checkpoint_metric=main_metric_name)

	if sched is not None:
		validator.add_callback_on_end(sched)

	print("Dataset : {:s} (train={:d} (supervised={:d}, unsupervised={:d}), val={:d}, eval={:s}).".format(
		args.dataset_name,
		len(dataset_train_augm_weak),
		len(dataset_train_s_augm_weak),
		len(dataset_train_u_augm_weak),
		len(dataset_val),
		str(len(dataset_eval)) if dataset_eval is not None else "None"
	))
	print("\nStart {:s} training on {:s} with model \"{:s}\" and {:d} epochs ({:s})...".format(TRAIN_NAME, args.dataset_name, model_name, args.nb_epochs, args.tag))
	start_time = time()

	for epoch in range(args.nb_epochs):
		trainer.train(epoch)
		validator.val(epoch)

	duration = time() - start_time
	print("\nEnd {:s} training. (duration = {:.2f})".format(TRAIN_NAME, duration))

	if dataset_eval is not None and checkpoint is not None and checkpoint.is_saved():
		recorder.deactivate_auto_storage()
		checkpoint.load_best_state(model, None)
		loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size_s, shuffle=False, drop_last=False)
		validator = Validater(model, activation, loader_eval, metrics_val, recorder, name="eval")
		validator.val(0)

	# Save results
	save_results(dirpath_writer, args, recorder, interface.get_transforms(args), main_metric_name, start_date, folds_val)

	if not recorder.is_empty():
		best = recorder.get_best_epoch(main_metric_name)
		print("Metric : \"{:s}\"".format(main_metric_name))
		print("Best epoch : {:d}".format(best["best_epoch"]))
		print("Best mean : {:f}".format(best["best_mean"]))
		print("Best std : {:f}".format(best["best_std"]))
	else:
		best = {}

	return best


def main():
	# Initialisation
	start_time = time()
	start_date = get_datetime()

	args = create_args()
	args.start_date = start_date
	args.train_name = TRAIN_NAME

	reset_seed(args.seed)
	torch.autograd.set_detect_anomaly(args.debug_mode)
	torch.cuda.empty_cache()

	print("Start {:s}. (tag: \"{:s}\")".format(TRAIN_NAME, args.tag))
	print(" - start_date: {:s}".format(start_date))

	# Prepare dataloaders
	interface = get_dataset_interface(args.dataset_name)
	args.nb_classes = interface.get_nb_classes()

	# Run
	if args.cross_validation:
		cross_validation(run_mixmatch, args, start_date, interface, TRAIN_NAME)
	else:
		run_mixmatch(args, start_date, args.fold_val, interface)

	exec_time = time() - start_time
	print("")
	print("Program started at \"{:s}\" and terminated at \"{:s}\".".format(start_date, get_datetime()))
	print("Total execution time: {:.2f}s".format(exec_time))


if __name__ == "__main__":
	main()
