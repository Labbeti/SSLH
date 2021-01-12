"""
	Main script for running a Supervised training.
	WARNING : It use only 10% of the dataset by default (use "--supervised_ratio 1.0" argument for running on 100% of the dataset).
"""

import torch

from argparse import ArgumentParser, Namespace

from mlu.datasets.wrappers import TransformDataset
from mlu.metrics import CategoricalAccuracy, MetricWrapper, AveragePrecision, RocAuc, DPrime
from mlu.nn import CrossEntropyWithVectors, Max

from sslh.augments.get_pool import get_transform
from sslh.dataset.get_interface import DatasetInterface
from sslh.models.get_model import get_model
from sslh.supervised.trainer import SupervisedTrainer
from sslh.utils.args import post_process_args, check_args, add_common_args
from sslh.utils.misc import (
	build_optimizer, build_scheduler, build_tensorboard_writer, build_checkpoint, get_prefix, main_run
)
from sslh.utils.recorder.recorder import Recorder
from sslh.utils.save import save_results
from sslh.utils.types import str_to_optional_str
from sslh.validation.validater import Validater

from time import time
from typing import Dict, List, Optional


RUN_NAME = "SupervisedAugment"


def create_args() -> Namespace:
	parser = ArgumentParser()
	add_common_args(parser)

	group_su_augm = parser.add_argument_group(RUN_NAME)
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


def run_supervised_augment(
	args: Namespace,
	start_date: str,
	interface: DatasetInterface,
	folds_train: Optional[List[int]],
	folds_val: Optional[List[int]],
	device: torch.device,
) -> Dict[str, Dict[str, float]]:
	"""
		Run a Supervised training with augmented data.

		:param args: The argparse arguments fo the run.
		:param start_date: Date of the start of the run.
		:param folds_train: The folds used for training the model.
		:param folds_val: The folds used for validating the model.
		:param interface: The dataset interface used for training.
		:param device: The main Pytorch device to use.
		:return: A dictionary containing the min and max scores on all epochs.
	"""

	# Builds augmentations
	data_type = interface.get_data_type()
	transform_base = interface.get_base_transform()
	transform_none = get_transform(args.augm_train, args, data_type, transform_base)
	transform_val = get_transform(args.augm_val, args, data_type, transform_base)
	target_transform = interface.get_target_transform()

	dataset_train_raw = interface.get_dataset_train(args.dataset_path, folds=folds_train)
	dataset_val = interface.get_dataset_val(args.dataset_path, transform_val, target_transform, folds=folds_val)
	dataset_eval = interface.get_dataset_eval(args.dataset_path, transform_val, target_transform)

	def transform_none_label(item: tuple) -> tuple:
		data, label = item
		return transform_none(data), target_transform(label)

	dataset_train_augm_none = TransformDataset(
		dataset_train_raw, transform_none_label, index=None,
	)

	loader_train = interface.get_loaders_split(
		labeled_dataset=dataset_train_raw,
		ratios=[args.supervised_ratio],
		datasets=[dataset_train_augm_none],
		batch_sizes=[args.batch_size_s],
		drop_last_list=[True],
		num_workers_list=[8],
		target_transformed=False,
	)[0]
	loader_val = interface.get_loader_val(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=False)

	# Prepare model
	model = get_model(args.model, args, device=device)
	model_name = model.__class__.__name__
	optim = build_optimizer(args, model)
	activation = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)

	criterion = CrossEntropyWithVectors()

	sched = build_scheduler(args, optim)

	# Prepare metrics
	target_type = interface.get_target_type()
	if target_type == "monolabel":
		main_metric_name = "val/acc"
		metrics_train = {"train/acc": CategoricalAccuracy(dim=1)}
		metrics_val = {
			"val/acc": CategoricalAccuracy(dim=1),
			"val/ce": MetricWrapper(CrossEntropyWithVectors(dim=1)),
			"val/max": MetricWrapper(Max(dim=1), use_target=False, reduce_fn=torch.mean),
		}
	elif target_type == "multilabel":
		main_metric_name = "val/mAP"
		metrics_train = {"train/mAP": AveragePrecision(), "train/mAUC": RocAuc(), "train/dPrime": DPrime()}
		metrics_val = {"val/mAP": AveragePrecision(), "val/mAUC": RocAuc(), "val/dPrime": DPrime()}
	else:
		raise RuntimeError(f"Unknown target type \"{target_type}\".")

	# Prepare objects for saving data
	prefix = get_prefix(args, folds_val, interface, start_date, model_name, RUN_NAME)
	writer, dirpath_writer = build_tensorboard_writer(args, prefix)
	recorder = Recorder(writer)
	checkpoint = build_checkpoint(args, dirpath_writer, model, optim)

	# Start main training
	trainer = SupervisedTrainer(
		model, activation, optim, loader_train, metrics_train, recorder, criterion, device=device)

	validater = Validater(
		model, activation, loader_val, metrics_val, recorder,
		device=device,
		checkpoint=checkpoint,
		checkpoint_metric=main_metric_name
	)

	if sched is not None:
		validater.add_callback_on_end(sched)

	trainer.add_callback_on_end(recorder)
	validater.add_callback_on_end(recorder)

	print("Dataset : {:s} (train={:d}, val={:d}, eval={:s}).".format(
		args.dataset_name,
		len(dataset_train_augm_none),
		len(dataset_val),
		str(len(dataset_eval)) if dataset_eval is not None else "None"
	))
	print("\nStart {:s} training on {:s} with model \"{:s}\" and {:d} epochs ({:s})...".format(RUN_NAME, args.dataset_name, model_name, args.nb_epochs, args.tag))
	start_time = time()

	for epoch in range(args.nb_epochs):
		trainer.train(epoch)
		validater.val(epoch)
		print()

	print("\nEnd {:s} training. (duration = {:.2f})".format(RUN_NAME, time() - start_time))

	if dataset_eval is not None and checkpoint is not None and checkpoint.is_saved():
		recorder.set_storage(write_std=False, write_min_mean=False, write_max_mean=False)
		checkpoint.load_best_state(model, None)
		loader_eval = interface.get_loader_val(dataset_eval, batch_size=args.batch_size_s, drop_last=False, num_workers=0)
		validater = Validater(model, activation, loader_eval, metrics_val, recorder, name="eval")
		validater.val(0)

	# Save results
	save_results(dirpath_writer, args, recorder, {}, main_metric_name, start_date, folds_val, start_time)

	if main_metric_name in recorder.get_all_names():
		idx_min, min_, idx_max, max_ = recorder.get_min_max(main_metric_name)
		print(f"Metric : \"{main_metric_name}\"")
		print(f"Max mean : {max_} at epoch {idx_max}")
		print(f"Min mean : {min_} at epoch {idx_min}")

		best = {
			main_metric_name: {
				"max": max_,
				"idx_max": idx_max,
				"min": min_,
				"idx_min": idx_min,
			}
		}
	else:
		best = {}

	return best


def main():
	main_run(create_args, run_supervised_augment, RUN_NAME)


if __name__ == "__main__":
	main()
