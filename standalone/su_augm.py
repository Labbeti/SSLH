"""
	Main script for running a Supervised training.
	WARNING : It use only 10% of the dataset by default (use "--supervised_ratio 1.0" argument for running on 100% of the dataset).
"""

import torch

from argparse import ArgumentParser, Namespace

from mlu.datasets.wrappers import TransformDataset
from mlu.metrics import CategoricalAccuracy, MetricWrapper, FScore
from mlu.nn import CrossEntropyWithVectors, Max
from mlu.utils.misc import get_nb_parameters

from sslh.augments.get_augm import get_augment_by_name, add_builder_process_transform
from sslh.datasets.get_builder import DatasetBuilder
from sslh.models.get_model import get_model
from sslh.supervised.trainer import SupervisedTrainer
from sslh.utils.args import post_process_args, check_args, add_common_args
from sslh.utils.misc import (
	get_optimizer, get_scheduler, get_tensorboard_writer, get_checkpoint, get_prefix, main_run, get_activation, evaluate
)
from sslh.utils.recorder.recorder import Recorder
from sslh.utils.save import save_results_files
from sslh.utils.types import str_to_optional_str
from sslh.validation.validater import Validater

from time import time
from torch.nn import DataParallel
from typing import Dict, List, Optional


RUN_NAME = "SupervisedAugment"


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser = add_common_args(parser)

	group_su_augm = parser.add_argument_group(f"{RUN_NAME} args")

	group_su_augm.add_argument("--augm_train", type=str_to_optional_str, default=None,
		help="The augment used for training. (default: None)")

	group_su_augm.add_argument("--augm_val", type=str_to_optional_str, default=None,
		help="The augment used for validation. (default: None)")

	group_su_augm.add_argument("--ratio", "-p", type=float, default=1.0,
		help="Ratio of the augments. (default: 1.0)")

	# OcclusionSR (augmentation_utils) ---------------------------------------------------------------------------------
	group_su_augm.add_argument("--occlusion_max_size", type=float, default=1.0,
		help="Occlusion argument: max_size. (default: 1.0)")

	group_su_augm.add_argument("--occlusion_sampling_rate", type=int, default=22050,
		help="Occlusion argument: sampling_rate. (default: 22050)")

	# CutOutSpec (mlu) ---------------------------------------------------------------------------------------
	group_su_augm.add_argument("--cutout_width_scale", type=float, nargs=2, default=(0.1, 0.5),
		help="CutOutSpec argument: width_scale. (default: [0.1, 0.5])")

	group_su_augm.add_argument("--cutout_height_scale", type=float, nargs=2, default=(0.1, 0.5),
		help="CutOutSpec argument: height_scale. (default: [0.1, 0.5])")

	# RandomTimeDropout (augmentation_utils) ---------------------------------------------------------------------------
	group_su_augm.add_argument("--random_time_dropout", type=float, default=0.01,
		help="RandomTimeDropout argument: dropout. (default: 0.01)")

	# RandomFreqDropout (augmentation_utils) ---------------------------------------------------------------------------
	group_su_augm.add_argument("--random_freq_dropout", type=float, default=0.01,
		help="RandomFreqDropout argument: dropout. (default: 0.01)")

	# Noise (NoiseSpec) (augmentation_utils) ---------------------------------------------------------------------------
	group_su_augm.add_argument("--noise_snr", type=float, default=10.0,
		help="NoiseSpec argument: snr. (default: 10.0)")

	# TimeStretch (augmentation_utils) ---------------------------------------------------------------------------------
	group_su_augm.add_argument("--time_stretch_rate", type=float, nargs=2, default=(0.9, 1.1),
		help="TimeStretch argument: rate. (default: [0.9, 1.1])")

	# StretchPadCrop (mlu) ---------------------------------------------------------------------------------------
	group_su_augm.add_argument("--stretchpadcrop_rate", type=float, nargs=2, default=(0.9, 1.1),
		help="StretchPadCrop argument: rate. (default: [0.9, 1.1])")

	group_su_augm.add_argument("--stretchpadcrop_align", type=str, default="left",
		help="StretchPadCrop argument: align. (default: left)")

	# PitchShiftRandom (augmentation_utils) ----------------------------------------------------------------------------
	group_su_augm.add_argument("--psr_sampling_rate", type=int, default=22050,
		help="PitchShiftRandom argument: sampling_rate. (default: 22050)")

	group_su_augm.add_argument("--psr_steps", type=int, nargs=2, default=(-3, 3),
		help="PitchShiftRandom argument: steps. (default: (-3, 3))")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	return args


def run_supervised_augment(
	args: Namespace,
	start_date: str,
	git_hash: str,
	builder: DatasetBuilder,
	folds_train: Optional[List[int]],
	folds_val: Optional[List[int]],
	device: torch.device,
) -> Dict[str, Dict[str, float]]:
	"""
		Run a Supervised training with augmented data.

		:param args: The argparse arguments fo the run.
		:param start_date: Date of the start of the run.
		:param git_hash: The current git hash of the repository.
		:param folds_train: The folds used for training the model.
		:param folds_val: The folds used for validating the model.
		:param builder: The dataset builder used for training.
		:param device: The main Pytorch device to use.
		:return: A dictionary containing the min and max scores on all epochs.
	"""

	# Builds augmentations
	augm_train = get_augment_by_name(args.augm_train, args)
	augm_val = get_augment_by_name(args.augm_val, args)

	transform_train = add_builder_process_transform(augm_train, builder)
	transform_val = add_builder_process_transform(augm_val, builder)
	target_transform = builder.get_target_transform()

	dataset_train_raw = builder.get_dataset_train(args.dataset_path, folds=folds_train, version=args.train_version)
	dataset_val = builder.get_dataset_val(args.dataset_path, transform_val, target_transform, folds=folds_val)
	dataset_eval = builder.get_dataset_eval(args.dataset_path, transform_val, target_transform)

	def transform_none_label(item: tuple) -> tuple:
		data, label = item
		return transform_train(data), target_transform(label)

	dataset_train_augm_none = TransformDataset(
		dataset_train_raw, transform_none_label, index=None,
	)

	loader_train = builder.get_loaders_split(
		labeled_dataset=dataset_train_raw,
		ratios=[args.supervised_ratio],
		datasets=[dataset_train_augm_none],
		batch_sizes=[args.batch_size_s],
		drop_last_list=[True],
		num_workers_list=[8],
		target_transformed=False,
	)[0]
	loader_val = builder.get_loader_val(dataset_val, args.batch_size_s)

	# Prepare model
	model = get_model(args.model, args, builder, device)
	if args.nb_gpu > 1:
		model = DataParallel(model)
	optim = get_optimizer(args, model)
	activation = get_activation(args.activation, clamp=True, clamp_min=2e-30)

	criterion = CrossEntropyWithVectors()

	sched = get_scheduler(args, optim)

	# Prepare metrics
	target_type = builder.get_target_type()
	if target_type == "monolabel":
		main_metric_name = "val/acc"

		metrics_train = {"train/acc": CategoricalAccuracy(dim=1)}
		metrics_val = {
			"val/acc": CategoricalAccuracy(dim=1),
			"val/ce": MetricWrapper(CrossEntropyWithVectors(dim=1)),
			"val/max": MetricWrapper(Max(dim=1), use_target=False, reduce_fn=torch.mean),
		}

	elif target_type == "multilabel":
		main_metric_name = "val/fscore"

		metrics_train = {"train/fscore": FScore()}
		metrics_val = {"val/fscore": FScore()}

	else:
		raise RuntimeError(f"Unknown target type '{target_type}'.")

	# Prepare objects for saving data
	prefix = get_prefix(args, folds_val, builder, start_date, args.model, RUN_NAME)
	writer, dirpath_writer = get_tensorboard_writer(args, prefix)
	recorder = Recorder(writer)
	checkpoint = get_checkpoint(args, dirpath_writer, model, optim)

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

	print("Dataset : {:s} (train={:d}, val={:d}, eval={:s}, folds_train={:s}, folds_val={:s}).".format(
		args.dataset_name,
		len(dataset_train_raw),
		len(dataset_val),
		str(len(dataset_eval)) if dataset_eval is not None else "None",
		str(folds_train),
		str(folds_val)
	))
	print("Model: {:s} ({:d} parameters).".format(args.model, get_nb_parameters(model)))
	print("\nStart {:s} training with {:d} epochs (tag: '{:s}')...".format(RUN_NAME, args.nb_epochs, args.tag))
	start_time = time()

	for epoch in range(args.nb_epochs):
		trainer.train(epoch)
		validater.val(epoch)
		print()

	duration = time() - start_time
	print("\nEnd {:s} training. (duration = {:.2f}s)".format(RUN_NAME, duration))

	evaluate(args, model, activation, builder, checkpoint, recorder, dataset_val, dataset_eval)

	# Save results
	save_results_files(dirpath_writer, RUN_NAME, duration, start_date, git_hash, folds_train, folds_val, builder, args, recorder)

	return recorder.get_all_min_max()


def main():
	main_run(create_args, run_supervised_augment, RUN_NAME)


if __name__ == "__main__":
	main()
