"""
	Main script for running a Supervised training.
	WARNING : It use only 10% of the dataset by default (use "--supervised_ratio 1.0" argument for running on 100% of the dataset).
"""

import torch

from argparse import ArgumentParser, Namespace
from augmentation_utils.augmentations import SignalAugmentation

from mlu.utils.misc import get_datetime, reset_seed

from ssl.augments.get_augm import get_augment_by_name
from ssl.datasets.get_interface import get_dataset_interface, DatasetInterface
from ssl.supervised.trainer import SupervisedTrainer
from ssl.utils.args import post_process_args, check_args, add_common_args
from ssl.utils.cross_validation import cross_validation
from ssl.utils.misc import build_optimizer, build_scheduler, build_tensorboard_writer, build_checkpoint, get_prefix
from ssl.utils.other_metrics import CategoricalAccuracyOnehot, CrossEntropyMetric, EntropyMetric, MaxMetric
from ssl.utils.recorder.recorder import Recorder
from ssl.utils.save import save_results
from mlu.nn import CrossEntropyWithVectors
from ssl.utils.types import str_to_optional_str
from ssl.validation.validater import Validater

from time import time
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose
from typing import Optional, Dict, Union


TRAIN_NAME = "SupervisedAugment"


def create_args() -> Namespace:
	parser = ArgumentParser()
	add_common_args(parser)

	group_su_augm = parser.add_argument_group(TRAIN_NAME)
	group_su_augm.add_argument("--augm_train", type=str_to_optional_str, default=None,
		help="The augment used for training. (default: None)")
	group_su_augm.add_argument("--augm_val", type=str_to_optional_str, default=None,
		help="The augment used for validation. (default: None)")

	group_su_augm.add_argument("--ratio", type=float, default=1.0,
		help="Ratio of the augments. (default: 1.0)")

	# Occlusion
	group_su_augm.add_argument("--occlusion_max_size", type=float, default=1.0,
		help="Occlusion argument: max_size. (default: 1.0)")

	# CutOutSpec
	group_su_augm.add_argument("--cutout_width_scale", type=float, nargs=2, default=(0.1, 0.5),
		help="CutOutSpec argument: width_scale. (default: [0.1, 0.5])")
	group_su_augm.add_argument("--cutout_height_scale", type=float, nargs=2, default=(0.1, 0.5),
		help="CutOutSpec argument: height_scale. (default: [0.1, 0.5])")

	# RandomTimeDropout
	group_su_augm.add_argument("--random_time_dropout", type=float, default=0.01,
		help="RandomTimeDropout argument: dropout. (default: 0.01)")

	# RandomFreqDropout
	group_su_augm.add_argument("--random_freq_dropout", type=float, default=0.01,
		help="RandomFreqDropout argument: dropout. (default: 0.01)")

	# Noise (NoiseSpec)
	group_su_augm.add_argument("--noise_snr", type=float, default=10.0,
		help="NoiseSpec argument: snr. (default: 10.0)")

	# TimeStretch
	group_su_augm.add_argument("--time_stretch_rate", type=float, nargs=2, default=(0.9, 1.1),
		help="TimeStretch argument: rate. (default: [0.9, 1.1])")

	# ResizePadCut
	group_su_augm.add_argument("--resize_rate", type=float, nargs=2, default=(0.9, 1.1),
		help="ResizePadCut argument: rate. (default: [0.9, 1.1])")
	group_su_augm.add_argument("--resize_align", type=str, default="left",
		help="ResizePadCut argument: align. (default: left)")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	return args


def run_supervised_augment(args: Namespace, start_date: str, fold_val: Optional[int], interface: DatasetInterface) -> Dict[str, Union[float, int]]:
	# Build loaders
	if interface.get_nb_folds() is None:
		folds_train, folds_val = None, None
	else:
		if fold_val is None:
			fold_val = interface.get_nb_folds()
		folds_train = list(range(1, interface.get_nb_folds() + 1))
		folds_train.remove(fold_val)
		folds_val = [fold_val]

	transform_augm_none = interface.get_transform_augm_none(args)

	transform_train = get_augment_by_name(args.augm_train, args)
	if isinstance(transform_train, SignalAugmentation):
		transform_train = Compose([transform_train, transform_augm_none])
	else:
		transform_train = Compose([transform_augm_none, transform_train])

	transform_val = get_augment_by_name(args.augm_val, args)
	if isinstance(transform_val, SignalAugmentation):
		transform_val = Compose([transform_val, transform_augm_none])
	else:
		transform_val = Compose([transform_augm_none, transform_val])

	dataset_train = interface.get_dataset_train_with_transform(args, folds_train, transform_train)
	dataset_val = interface.get_dataset_val_with_transform(args, folds_val, transform_val)
	dataset_eval = interface.get_dataset_eval_with_transform(args, None, transform_val)

	if args.supervised_ratio < 1.0:
		indexes = interface.generate_indexes_for_split(dataset_train, [args.supervised_ratio])[0]
		dataset_train = Subset(dataset_train, indexes)

	loader_train = DataLoader(dataset_train, batch_size=args.batch_size_s, shuffle=True, num_workers=8, drop_last=True)
	loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=False)

	# Prepare model
	model = interface.build_model(args.model, args)
	model_name = model.__class__.__name__
	optim = build_optimizer(args, model)
	activation = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)

	criterion = CrossEntropyWithVectors()

	sched = build_scheduler(args, optim)

	# Prepare metrics
	main_metric_name = "val/acc"
	metrics_train = {"acc": CategoricalAccuracyOnehot(dim=1)}
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
	trainer = SupervisedTrainer(model, activation, optim, loader_train, metrics_train, recorder, criterion)
	validator = Validater(model, activation, loader_val, metrics_val, recorder, checkpoint=checkpoint, checkpoint_metric=main_metric_name)

	if sched is not None:
		validator.add_callback_on_end(sched)

	print("Dataset : {:s} (train={:d}, val={:d}, eval={:s}).".format(
		args.dataset_name,
		len(dataset_train),
		len(dataset_val),
		str(len(dataset_eval)) if dataset_eval is not None else "None"
	))
	print("\nStart {:s} training on {:s} with model \"{:s}\" and {:d} epochs ({:s})...".format(TRAIN_NAME, args.dataset_name, model_name, args.nb_epochs, args.tag))
	start_time = time()

	for epoch in range(args.nb_epochs):
		trainer.train(epoch)
		validator.val(epoch)

	print("\nEnd {:s} training. (duration = {:.2f})".format(TRAIN_NAME, time() - start_time))

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
		cross_validation(run_supervised_augment, args, start_date, interface, TRAIN_NAME)
	else:
		run_supervised_augment(args, start_date, args.fold_val, interface)

	exec_time = time() - start_time
	print("")
	print("Program started at \"{:s}\" and terminated at \"{:s}\".".format(start_date, get_datetime()))
	print("Total execution time: {:.2f}s".format(exec_time))


if __name__ == "__main__":
	main()
