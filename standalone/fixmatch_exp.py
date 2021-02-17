"""
	Main script for running a FixMatch Experimental training.
	This file contains more options in order to test FixMatch variants.
"""

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import torch

from argparse import ArgumentParser, Namespace

from mlu.datasets.wrappers import TransformDataset
from mlu.metrics import CategoricalAccuracy, MetricWrapper, FScore
from mlu.nn import CrossEntropyWithVectors, Max, JSDivLoss, KLDivLossWithProbabilities, BCELossBatchMean
from mlu.utils.misc import get_nb_parameters
from mlu.utils.zip_cycle import ZipCycle

from sslh.augments.get_pool import get_transform
from sslh.datasets.get_builder import DatasetBuilder
from sslh.fixmatch.loss import FixMatchLoss, FixMatchLossSoftReduceU
from sslh.fixmatch.trainer import FixMatchTrainer
from sslh.fixmatch.trainer_adv import FixMatchTrainerAdv
from sslh.fixmatch.trainer_mixup import FixMatchTrainerMixUp
from sslh.fixmatch.trainer_mixup_shuffle import FixMatchTrainerMixUpShuffle
from sslh.fixmatch.trainer_mixup_teacher import FixMatchTrainerMixUpTeacher
from sslh.fixmatch.trainer_teacher import FixMatchTrainerTeacher
from sslh.fixmatch.trainer_teacher_label_u import FixMatchTrainerTeacherLabelU
from sslh.fixmatch.trainer_threshold_guess import FixMatchTrainerThresholdGuess
from sslh.fixmatch.trainer_uniloss import FixMatchTrainerUniLoss
from sslh.mixmatch.warmup import WarmUp
from sslh.models.get_model import get_model
from sslh.utils.args import post_process_args, check_args, add_common_args
from sslh.utils.misc import (
	get_optimizer, get_scheduler, get_tensorboard_writer, get_checkpoint, get_prefix, main_run, get_activation, evaluate
)
from sslh.utils.recorder.recorder import Recorder
from sslh.utils.save import save_results_files
from sslh.utils.types import str_to_bool, str_to_optional_str
from sslh.validation.validater import Validater

from time import time
from torch.nn import MSELoss, DataParallel
from typing import Dict, List, Optional


RUN_NAME = "FixMatchExp"


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser = add_common_args(parser)

	group_fm = parser.add_argument_group(f"{RUN_NAME} args")

	group_fm.add_argument("--lambda_u", type=float, default=1.0,
		help="MixMatch, FixMatch and ReMixMatch 'lambda_u' hyperparameter. "
		"Coefficient of unsupervised loss component. (default: 1.0)")

	group_fm.add_argument("--batch_size_u", "--bsize_u", "--bu", type=int, default=30,
		help="Batch size used for unsupervised loader. (default: 128)")

	group_fm.add_argument("--criterion_s", type=str, default="ce",
		choices=["mse", "ce", "kl", "js", "bce"],
		help="FixMatch supervised loss component. (default: 'ce')")

	group_fm.add_argument("--criterion_u", type=str, default="ce",
		choices=["mse", "ce", "kl", "js", "bce"],
		help="FixMatch unsupervised loss component. (default: 'ce')")

	group_fm.add_argument("--threshold", "--threshold_confidence", type=float, default=0.95,
		help="FixMatch threshold for compute confidence mask in loss. (default: 0.95)")

	group_fm.add_argument("--use_mixup", "--mixup", type=str_to_bool, default=False,
		help="Apply MixUp between supervised and unsupervised data. (default: False)")

	group_fm.add_argument("--alpha", "--mixup_alpha", type=float, default=0.75,
		help="Alpha hyperparameter used in MixUp. (default: 0.75)")

	group_fm.add_argument("--use_uniloss", type=str_to_bool, default=False,
		help="Use experimental uniloss training. (default: False)")

	group_fm.add_argument("--start_probs", type=float, nargs=2, default=[1.0, 0.0],
		help="Probabilities to compute losses at start for Uniloss. (default: [1.0, 0.0])")

	group_fm.add_argument("--target_probs", type=float, nargs=2, default=[0.5, 0.5],
		help="Probabilities to compute losses at end for Uniloss. (default: [0.5, 0.5])")

	group_fm.add_argument("--use_adversarial", "--use_adv", type=str_to_bool, default=False,
		help="Use adversarial data instead of augmentations in MixMatch. (default: False)")

	group_fm.add_argument("--epsilon_adv_weak", type=float, default=1e-2,
		help="Epsilon used in FGSM adversarial method. (default: 1e-2)")

	group_fm.add_argument("--epsilon_adv_strong", type=float, default=1e-0,
		help="Epsilon used in FGSM adversarial method. (default: 1e-0)")

	group_fm.add_argument("--augm_none", type=str_to_optional_str, default=None,
		help="Augment pool for default training dataset. (default: None)")

	group_fm.add_argument("--augm_weak", type=str_to_optional_str, default="weak",
		help="Augment pool for weak augmentation to use. (default: 'weak')")

	group_fm.add_argument("--augm_strong", type=str_to_optional_str, default="strong",
		help="Augment pool for strong augmentation to use. (default: 'strong')")

	group_fm.add_argument("--use_mixup_shuffle", type=str_to_bool, default=False,
		help="Use MixUp between batch with itself shuffled (no mix between labeled and unlabeled data). (default: False)")

	group_fm.add_argument("--use_teacher", "--teacher", type=str_to_bool, default=False,
		help="Use a teacher for guessing labels. The teacher is updated with an EMA. (default: False)")

	group_fm.add_argument("--ema_decay", "--decay", type=float, default=0.999,
		help="EMA decay used by FixMatch-Teacher. (default: 0.999)")

	group_fm.add_argument("--use_teacher_true_label_u", type=str_to_bool, default=False,
		help="Use a teacher for guessing labels. The teacher is updated with an EMA. (default: False)")

	group_fm.add_argument("--use_warmup_by_iteration", type=str_to_bool, default=False,
		help="Use WarmUp on lambda_u hparam every epochs. Compatible with most of the training variants. (default: False)")

	group_fm.add_argument("--warmup_nb_epochs", type=int, default=10,
		help="The number of epochs before warmup reach the lambda_u max value. (default: 10)")

	group_fm.add_argument("--use_mixup_teacher", type=str_to_bool, default=False,
		help="Combine MixUp and Teacher with FixMatch. (default: False)")

	group_fm.add_argument("--use_soft_reduce_u", type=str_to_bool, default=False,
		help="Activate soft reduce u for FixMatch loss, which means the unsupervised loss component is mean reduced "
			"using the number of pseudo labels used instead of the bsize_u. (default: False)")

	group_fm.add_argument("--use_threshold_guess", type=str_to_bool, default=False,
		help="Use a threshold for binarize label in fixmatch. Use --threshold_guess option for control this threshold. "
			"(default: False)")

	group_fm.add_argument("--threshold_guess", type=float, default=0.5,
		help="Threshold used for binarize guessed label in fixmatch. Useful for replace argmax() when targets are multihot. "
			"(default: 0.5)")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	if args.use_mixup and args.use_teacher and not args.use_mixup_teacher:
		args.use_mixup = False
		args.use_teacher = False
		args.use_mixup_teacher = True

	return args


def run_fixmatch_exp(
	args: Namespace,
	start_date: str,
	git_hash: str,
	builder: DatasetBuilder,
	folds_train: Optional[List[int]],
	folds_val: Optional[List[int]],
	device: torch.device,
) -> Dict[str, Dict[str, float]]:
	"""
		Run a FixMatch Experimental training.

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
	transform_weak = get_transform(args.augm_weak, args, builder)
	transform_strong = get_transform(args.augm_strong, args, builder)
	transform_val = get_transform("identity", args, builder)
	target_transform = builder.get_target_transform()

	dataset_train_raw = builder.get_dataset_train(args.dataset_path, folds=folds_train, version=args.train_version)
	dataset_val = builder.get_dataset_val(args.dataset_path, transform_val, target_transform, folds=folds_val)
	dataset_eval = builder.get_dataset_eval(args.dataset_path, transform_val, target_transform)

	if not args.use_adversarial:
		def transform_weak_label(item: tuple) -> tuple:
			data, label = item
			return transform_weak(data), target_transform(label)

		def transform_weak_strong(item: tuple) -> tuple:
			data, label = item
			return transform_weak(data), transform_strong(data), target_transform(label)

		def transform_weak_strong_no_label(item: tuple) -> tuple:
			data, label = item
			return transform_weak(data), transform_strong(data)

		if not args.use_teacher_true_label_u:
			transform_weak_strong_ = transform_weak_strong_no_label
		else:
			transform_weak_strong_ = transform_weak_strong

		dataset_train_augm_weak = TransformDataset(
			dataset_train_raw, transform_weak_label, index=None)
		dataset_train_augm_weak_strong_no_label = TransformDataset(
			dataset_train_raw, transform_weak_strong_, index=None)

		loader_train_s, loader_train_u = builder.get_loaders_split(
			labeled_dataset=dataset_train_raw,
			ratios=[args.supervised_ratio, 1.0 - args.supervised_ratio],
			datasets=[dataset_train_augm_weak, dataset_train_augm_weak_strong_no_label],
			batch_sizes=[args.batch_size_s, args.batch_size_u],
			drop_last_list=[True, True],
			num_workers_list=[2, 6],
			target_transformed=False,
		)
	else:
		def transform_none_label(item: tuple) -> tuple:
			data, label = item
			return transform_none(data), target_transform(label)

		dataset_train_augm_none = TransformDataset(
			dataset_train_raw, transform_none_label, index=None)

		loader_train_s, loader_train_u = builder.get_loaders_split(
			labeled_dataset=dataset_train_raw,
			ratios=[args.supervised_ratio, 1.0 - args.supervised_ratio],
			datasets=[dataset_train_augm_none, dataset_train_augm_none],
			batch_sizes=[args.batch_size_s, args.batch_size_u],
			drop_last_list=[True, True],
			num_workers_list=[2, 6],
			target_transformed=False,
		)

	loader_train = ZipCycle([loader_train_s, loader_train_u], policy=args.zip_cycle_policy)
	loader_val = builder.get_loader_val(dataset_val, args.batch_size_s)

	# Prepare model
	model = get_model(args.model, args, builder, device)
	if args.nb_gpu > 1:
		model = DataParallel(model)
	optim = get_optimizer(args, model)
	activation = get_activation(args.activation, clamp=True, clamp_min=2e-30)

	loss_mapper = {
		"mse": MSELoss(reduction="none"),
		"ce": CrossEntropyWithVectors(reduction="none"),
		"kl": KLDivLossWithProbabilities(reduction="none"),
		"js": JSDivLoss(reduction="none"),
		"bce": BCELossBatchMean(),
	}
	criterion_s = loss_mapper[args.criterion_s]
	criterion_u = loss_mapper[args.criterion_u]
	if not args.use_soft_reduce_u:
		criterion = FixMatchLoss(criterion_s, criterion_u)
	else:
		criterion = FixMatchLossSoftReduceU(criterion_s, criterion_u)

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
		main_metric_name = "val/fscore"

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
		trainer = FixMatchTrainerMixUp(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			device=device,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			alpha=args.alpha,
		)
	elif args.use_uniloss:
		trainer = FixMatchTrainerUniLoss(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, device=device,
			nb_epochs=args.nb_epochs,
			criterion=criterion,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			start_probs=args.start_probs,
			target_probs=args.target_probs,
		)
	elif args.use_adversarial:
		trainer = FixMatchTrainerAdv(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			device=device,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			criterion_adv=CrossEntropyWithVectors(),
			epsilon_adv_weak=args.epsilon_adv_weak,
			epsilon_adv_strong=args.epsilon_adv_strong,
		)
	elif args.use_mixup_shuffle:
		trainer = FixMatchTrainerMixUpShuffle(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			device=device,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			alpha=args.alpha,
		)
	elif args.use_teacher:
		trainer = FixMatchTrainerTeacher(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			device=device,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			decay=args.ema_decay,
		)
	elif args.use_teacher_true_label_u:
		trainer = FixMatchTrainerTeacherLabelU(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			device=device,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			decay=args.ema_decay,
		)
	elif args.use_mixup_teacher:
		trainer = FixMatchTrainerMixUpTeacher(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			device=device,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			alpha=args.alpha,
			decay=args.ema_decay,
		)
	elif args.use_threshold_guess:
		trainer = FixMatchTrainerThresholdGuess(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			device=device,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			threshold_guess=args.threshold_guess,
		)
	else:
		trainer = FixMatchTrainer(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			device=device,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
		)

	validater = Validater(
		model, activation, loader_val, metrics_val, recorder,
		device=device,
		checkpoint=checkpoint,
		checkpoint_metric=main_metric_name
	)

	if sched is not None:
		validater.add_callback_on_end(sched)

	if args.use_warmup_by_iteration:
		warmup = WarmUp(max_value=args.lambda_u, nb_steps=args.warmup_nb_epochs, obj=trainer, attr_name="lambda_u")
		validater.add_callback_on_end(warmup)

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
	main_run(create_args, run_fixmatch_exp, RUN_NAME)


if __name__ == "__main__":
	main()
