"""
	Main script for running a Supervised Experimental training.
	WARNING : It use only 10% of the dataset by default (use "--supervised_ratio 1.0" argument for running on 100% of the dataset).
	This file contains more options in order to test Supervised variants.
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

from sslh.augments.get_pool import get_transform
from sslh.datasets.get_builder import DatasetBuilder
from sslh.models.get_model import get_model
from sslh.supervised.loss import MixUpLoss, MixUpLossSmooth
from sslh.supervised.trainer import SupervisedTrainer
from sslh.supervised.trainer_acc import SupervisedTrainerAcc
from sslh.supervised.trainer_adv import SupervisedTrainerAdv
from sslh.supervised.trainer_cutmix import SupervisedTrainerCutMix
from sslh.supervised.trainer_cutmixspec import SupervisedTrainerCutMixSpec
from sslh.supervised.trainer_mixup import SupervisedTrainerMixUp
from sslh.supervised.trainer_mixup_mix_label import SupervisedTrainerMixUpMixLabel
from sslh.supervised.trainer_mixup_mix_label_sharp import SupervisedTrainerMixUpMixLabelSharp
from sslh.supervised.trainer_mixup_roll import SupervisedMixUpRollTrainer
from sslh.supervised.trainer_mixup_uniform import SupervisedTrainerMixUpUniform
from sslh.utils.args import post_process_args, check_args, add_common_args
from sslh.utils.misc import (
	get_optimizer, get_scheduler, get_tensorboard_writer, get_checkpoint, get_prefix, main_run, get_activation, evaluate
)
from sslh.utils.recorder.recorder import Recorder
from sslh.utils.save import save_results_files
from sslh.utils.types import str_to_optional_str, str_to_bool
from sslh.validation.validater import Validater

from time import time
from torch.nn import MSELoss, BCELoss, DataParallel
from typing import Dict, List, Optional


RUN_NAME = "SupervisedExp"


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser = add_common_args(parser)

	group_su = parser.add_argument_group(f"{RUN_NAME} args")

	group_su.add_argument("--augment", "--augment_type", type=str_to_optional_str, default=None,
		choices=[None, "weak", "strong"],
		help="Apply identity, weak or strong augment on supervised train dataset. (default: None)")

	group_su.add_argument("--backward_frequency", type=int, default=1,
		help="Activate accumulative loss in order to update model not on every iteration. (default: 1)")

	group_su.add_argument("--criterion", type=str, default="ce",
		choices=["mse", "ce", "kl", "js", "bce"],
		help="Supervised loss. (default: 'ce')")

	group_su.add_argument("--use_mixup", type=str_to_bool, default=False,
		help="Apply MixUp between supervised and unsupervised data. (default: False)")

	group_su.add_argument("--alpha", "--mixup_alpha", type=float, default=0.4,
		help="Alpha hyperparameter used in MixUp. (default: 0.4)")

	group_su.add_argument("--use_adversarial", "--use_adv", type=str_to_bool, default=False,
		help="Use adversarial data instead of augmentations in MixMatch. (default: False)")

	group_su.add_argument("--epsilon_adv", type=float, default=1e-2,
		help="Epsilon used in FGSM adversarial method. (default: 1e-2)")

	group_su.add_argument("--augm_none", type=str_to_optional_str, default=None,
		help="Augment pool for default training dataset. (default: None)")

	group_su.add_argument("--use_log_softmax", type=str_to_bool, default=False,
		help="Use LogSoftmax as activation layer and change criterion for having log_input. (default: False)")

	group_su.add_argument("--use_mixup_mix_label", type=str_to_bool, default=False,
		help="Use MixUp with mixed labels like in MixMatch. (default: False)")

	group_su.add_argument("--use_mixup_smooth", type=str_to_bool, default=False,
		help="Use MixUp with smoothed labels. (default: False)")

	group_su.add_argument("--use_mixup_mix_label_sharp", type=str_to_bool, default=False,
		help="Use MixUp with mixed labels and sharp labels. (default: False)")

	group_su.add_argument("--sharp_temperature", type=float, default=0.3,
		help="Label sharp temperature with MixUp mix labels. (default: 0.3)")

	group_su.add_argument("--use_mixup_uniform", type=str_to_bool, default=False,
		help="Use MixUp with uniform mask.")

	group_su.add_argument("--use_cutmix", type=str_to_bool, default=False,
		help="Use CutMix training.")

	group_su.add_argument("--use_cutmixspec", type=str_to_bool, default=False,
		help="Use CutMixSpec training.")

	group_su.add_argument("--use_mixup_roll", type=str_to_bool, default=False,
		help="Roll spectrogram along the last dimension before MixUp.")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	return args


def run_supervised_exp(
	args: Namespace,
	start_date: str,
	git_hash: str,
	builder: DatasetBuilder,
	folds_train: Optional[List[int]],
	folds_val: Optional[List[int]],
	device: torch.device,
) -> Dict[str, Dict[str, float]]:
	"""
		Run a Supervised Experimental training.

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
	if not args.use_log_softmax:
		activation = get_activation(args.activation, clamp=True, clamp_min=2e-30)
	else:
		activation = lambda x, dim: x.log_softmax(dim=dim)
		if args.criterion not in ["ce", "kl"]:
			raise RuntimeError("Invalid criterion '{:s}' with log_softmax.".format(args.criterion))

	loss_mapper = {
		"mse": MSELoss(),
		"ce": CrossEntropyWithVectors(log_input=args.use_log_softmax),
		"kl": KLDivLossWithProbabilities(log_input=args.use_log_softmax),
		"js": JSDivLoss(),
		"bce": BCELoss(),
	}
	criterion = loss_mapper[args.criterion]
	if args.use_mixup or args.use_mixup_uniform or args.use_cutmix or args.use_cutmixspec or args.use_mixup_roll:
		criterion = MixUpLoss(criterion)
	elif args.use_mixup_smooth:
		criterion = MixUpLossSmooth(criterion)

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
	if args.backward_frequency > 1:
		trainer = SupervisedTrainerAcc(
			model, activation, optim, loader_train, metrics_train, recorder, criterion,
			backward_frequency=args.backward_frequency)
	elif args.use_mixup or args.use_mixup_smooth:
		trainer = SupervisedTrainerMixUp(
			model, activation, optim, loader_train, metrics_train, recorder, criterion,
			alpha=args.alpha)
	elif args.use_adversarial:
		trainer = SupervisedTrainerAdv(
			model, activation, optim, loader_train, metrics_train, recorder, criterion,
			criterion_adv=CrossEntropyWithVectors(), epsilon_adv=args.epsilon_adv)
	elif args.use_mixup_mix_label:
		trainer = SupervisedTrainerMixUpMixLabel(
			model, activation, optim, loader_train, metrics_train, recorder, criterion,
			alpha=args.alpha)
	elif args.use_mixup_mix_label_sharp:
		trainer = SupervisedTrainerMixUpMixLabelSharp(
			model, activation, optim, loader_train, metrics_train, recorder, criterion,
			alpha=args.alpha, temperature=args.sharp_temperature)
	elif args.use_mixup_uniform:
		trainer = SupervisedTrainerMixUpUniform(
			model, activation, optim, loader_train, metrics_train, recorder, criterion)
	elif args.use_cutmix:
		trainer = SupervisedTrainerCutMix(
			model, activation, optim, loader_train, metrics_train, recorder, criterion,
			alpha=args.alpha)
	elif args.use_cutmixspec:
		trainer = SupervisedTrainerCutMixSpec(
			model, activation, optim, loader_train, metrics_train, recorder, criterion,
			alpha=args.alpha)
	elif args.use_mixup_roll:
		trainer = SupervisedMixUpRollTrainer(
			model, activation, optim, loader_train, metrics_train, recorder, criterion,
			alpha=args.alpha, dim_rool=-1)
	else:
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
	main_run(create_args, run_supervised_exp, RUN_NAME)


if __name__ == "__main__":
	main()
