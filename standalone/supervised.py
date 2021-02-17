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

from sslh.augments.get_pool import get_transform
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
from torch.nn import BCELoss, DataParallel
from typing import Dict, List, Optional


RUN_NAME = "Supervised"


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser = add_common_args(parser)

	group_su = parser.add_argument_group(f"{RUN_NAME} args")

	group_su.add_argument("--augm_none", type=str_to_optional_str, default=None,
		help="Augment pool for default training dataset. (default: None)")

	group_su.add_argument("--criterion", type=str, default="ce",
		choices=["ce", "bce"],
		help="Supervised loss. (default: 'ce')")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	return args


def run_supervised(
	args: Namespace,
	start_date: str,
	git_hash: str,
	builder: DatasetBuilder,
	folds_train: Optional[List[int]],
	folds_val: Optional[List[int]],
	device: torch.device,
) -> Dict[str, Dict[str, float]]:
	"""
		Run a Supervised training.

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
	transform_none = get_transform(args.augm_none, args, builder)
	transform_val = get_transform("identity", args, builder)
	target_transform = builder.get_target_transform()

	dataset_train_raw = builder.get_dataset_train(args.dataset_path, folds=folds_train, version=args.train_version)
	dataset_val = builder.get_dataset_val(args.dataset_path, transform_val, target_transform, folds=folds_val)
	dataset_eval = builder.get_dataset_eval(args.dataset_path, transform_val, target_transform)

	def transform_none_label(item: tuple) -> tuple:
		data, label = item
		return transform_none(data), target_transform(label)

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

	if args.criterion == "ce":
		criterion = CrossEntropyWithVectors()
	elif args.criterion == "bce":
		criterion = BCELoss()
	else:
		raise RuntimeError(f"Unknown criterion '{args.criterion}'.")

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
		main_metric_name = "val/mAP"

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
	trainer = SupervisedTrainer(model, activation, optim, loader_train, metrics_train, recorder, criterion, device=device)
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
	main_run(create_args, run_supervised, RUN_NAME)


if __name__ == "__main__":
	main()
