"""
	Main script for running a UDA Experimental training.
	This file contains more options in order to test UDA variants.
"""

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import torch

from argparse import ArgumentParser, Namespace

from mlu.datasets.wrappers import TransformDataset
from mlu.metrics import CategoricalAccuracy, MetricWrapper, FScore
from mlu.nn import CrossEntropyWithVectors, Max, JSDivLoss, KLDivLossWithProbabilities
from mlu.utils.misc import get_nb_parameters
from mlu.utils.zip_cycle import ZipCycle

from sslh.augments.get_pool import get_transform
from sslh.datasets.get_builder import DatasetBuilder
from sslh.models.get_model import get_model
from sslh.uda.loss import UDALoss
from sslh.uda.trainer import UDATrainer
from sslh.uda.trainer_mixup import UDATrainerMixUp
from sslh.utils.args import post_process_args, check_args, add_common_args
from sslh.utils.misc import (
	get_optimizer, get_scheduler, get_tensorboard_writer, get_checkpoint, get_prefix, main_run, get_activation, evaluate
)
from sslh.utils.recorder.recorder import Recorder
from sslh.utils.save import save_results_files
from sslh.utils.types import str_to_optional_str, str_to_bool
from sslh.validation.validater import Validater

from time import time
from torch.nn import MSELoss, DataParallel
from typing import Dict, List, Optional


RUN_NAME = "UDAExp"


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser = add_common_args(parser)

	group_uda = parser.add_argument_group(f"{RUN_NAME} args")

	group_uda.add_argument("--lambda_u", type=float, default=1.0,
		help="MixMatch, FixMatch and ReMixMatch 'lambda_u' hyperparameter. "
			 "Coefficient of unsupervised loss component. (default: 1.0)")

	group_uda.add_argument("--batch_size_u", "--bsize_u", "--bu", type=int, default=30,
		help="Batch size used for unsupervised loader. (default: 128)")

	group_uda.add_argument("--criterion_s", type=str, default="ce",
		choices=["mse", "ce", "kl", "js"],
		help="UDA supervised loss component. (default: 'ce')")

	group_uda.add_argument("--criterion_u", type=str, default="ce",
		choices=["mse", "ce", "kl", "js"],
		help="UDA unsupervised loss component. (default: 'ce')")

	group_uda.add_argument("--threshold", "--threshold_confidence", type=float, default=0.8,
		help="FixMatch threshold for compute confidence mask in loss. (default: 0.8)")

	group_uda.add_argument("--temperature", "--sharpen_temperature", type=float, default=0.4,
		help="MixMatch and ReMixMatch hyperparameter temperature used by sharpening. (default: 0.4)")

	group_uda.add_argument("--augment", "--augment_type", type=str_to_optional_str, default="strong",
		help="Augment type used in UDA training. (default: 'strong')")

	group_uda.add_argument("--use_mixup", type=str_to_bool, default=False,
		help="Apply MixUp between supervised and unsupervised data. (default: False)")

	group_uda.add_argument("--alpha", "--mixup_alpha", type=float, default=0.75,
		help="Alpha hyperparameter used in MixUp. (default: 0.75)")

	group_uda.add_argument("--augm_none", type=str_to_optional_str, default=None,
		help="Augment pool for default training dataset. (default: None)")

	group_uda.add_argument("--augm_strong", type=str_to_optional_str, default="strong",
		help="Augment pool for strong augmentation to use. (default: 'strong')")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	return args


def run_uda_exp(
	args: Namespace,
	start_date: str,
	git_hash: str,
	builder: DatasetBuilder,
	folds_train: Optional[List[int]],
	folds_val: Optional[List[int]],
	device: torch.device,
) -> Dict[str, Dict[str, float]]:
	"""
		Run a UDA training.

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
	transform_strong = get_transform(args.augm_strong, args, builder)
	transform_val = get_transform("identity", args, builder)
	target_transform = builder.get_target_transform()

	dataset_train_raw = builder.get_dataset_train(args.dataset_path, folds=folds_train, version=args.train_version)
	dataset_val = builder.get_dataset_val(args.dataset_path, transform_val, target_transform, folds=folds_val)
	dataset_eval = builder.get_dataset_eval(args.dataset_path, transform_val, target_transform, folds=folds_val)

	def transform_none_label(item: tuple) -> tuple:
		data, label = item
		return transform_none(data), target_transform(label)

	def transform_strong_no_label(item: tuple) -> tuple:
		data, label = item
		return data, transform_strong(data)

	dataset_train_augm_none = TransformDataset(
		dataset_train_raw, transform_none_label, index=None,
	)
	dataset_train_strong_no_label = TransformDataset(
		dataset_train_raw, transform_strong_no_label, index=None,
	)

	loader_train_s, loader_train_u = builder.get_loaders_split(
		labeled_dataset=dataset_train_raw,
		ratios=[args.supervised_ratio, 1.0 - args.supervised_ratio],
		datasets=[dataset_train_augm_none, dataset_train_strong_no_label],
		batch_sizes=[args.batch_size_s, args.batch_size_u],
		drop_last_list=[True, True],
		num_workers_list=[2, 6],
		target_transformed=False,
	)
	loader_train = ZipCycle([loader_train_s, loader_train_u], policy=args.zip_cycle_policy)
	loader_val = builder.get_loader_val(dataset_val, args.batch_size_s)

	# Prepare model
	model = get_model(args.model, args, builder, device=device)
	if args.nb_gpu > 1:
		model = DataParallel(model)
	optim = get_optimizer(args, model)
	activation = get_activation(args.activation, clamp=True, clamp_min=2e-30)

	loss_mapper = {
		"mse": MSELoss(reduction="none"),
		"ce": CrossEntropyWithVectors(reduction="none"),
		"kl": KLDivLossWithProbabilities(reduction="none"),
		"js": JSDivLoss(reduction="none"),
	}
	criterion_s = loss_mapper[args.criterion_s]
	criterion_u = loss_mapper[args.criterion_u]
	criterion = UDALoss(criterion_s, criterion_u)

	sched = get_scheduler(args, optim)

	# Prepare metrics
	target_type = builder.get_target_type()
	if target_type == "monolabel":
		main_metric_name = "val/acc"

		metrics_train_s = {"train/acc_s": CategoricalAccuracy(dim=1)}
		metrics_train_u = {"train/acc_u": CategoricalAccuracy(dim=1)}
		metrics_val = {
			"val/acc": CategoricalAccuracy(dim=1),
			"val/ce": MetricWrapper(CrossEntropyWithVectors(dim=1)),
			"val/max": MetricWrapper(Max(dim=1), use_target=False, reduce_fn=torch.mean),
		}

	elif target_type == "multilabel":
		main_metric_name = "val/mAP"
		metrics_train_s = {"train/fscore_s": FScore()}
		metrics_train_u = {"train/fscore_u": FScore()}
		metrics_val = {"val/fscore": FScore()}

	else:
		raise RuntimeError(f"Unknown target type '{target_type}'.")

	# Prepare objects for saving data
	prefix = get_prefix(args, folds_val, builder, start_date, args.model, RUN_NAME)
	writer, dirpath_writer = get_tensorboard_writer(args, prefix)
	recorder = Recorder(writer)
	checkpoint = get_checkpoint(args, dirpath_writer, model, optim)

	# Start main training
	if args.use_mixup:
		trainer = UDATrainerMixUp(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			temperature=args.temperature,
			alpha=args.alpha,
		)
	else:
		trainer = UDATrainer(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			device=device,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			temperature=args.temperature,
		)
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
	main_run(create_args, run_uda_exp, RUN_NAME)


if __name__ == "__main__":
	main()
