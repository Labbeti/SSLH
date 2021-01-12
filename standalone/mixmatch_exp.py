"""
	Main script for running a MixMatch Experimental training.
	This file contains more options in order to test MixMatch variants.
"""

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import torch

from argparse import ArgumentParser, Namespace

from mlu.datasets.wrappers import TransformDataset
from mlu.metrics import CategoricalAccuracy, MetricWrapper, AveragePrecision, RocAuc, DPrime
from mlu.nn import CrossEntropyWithVectors, Max, JSDivLoss, KLDivLossWithProbabilities
from mlu.utils.zip_cycle import ZipCycle

from sslh.augments.get_pool import get_transform
from sslh.dataset.get_interface import DatasetInterface
from sslh.mixmatch.loss import MixMatchLoss, MixMatchLossNoLabelMix
from sslh.mixmatch.trainer import MixMatchTrainer
from sslh.mixmatch.trainer_acc import MixMatchTrainerAcc
from sslh.mixmatch.trainer_adv import MixMatchTrainerAdv
from sslh.mixmatch.trainer_argmax import MixMatchTrainerArgmax
from sslh.mixmatch.trainer_no_label_mix import MixMatchTrainerNoLabelMix
from sslh.mixmatch.trainer_no_warmup import MixMatchTrainerNoWarmUp
from sslh.mixmatch.trainer_true_label import MixMatchTrainerTrueLabel
from sslh.mixmatch.warmup import WarmUp
from sslh.models.get_model import get_model

from sslh.utils.args import post_process_args, check_args, add_common_args
from sslh.utils.misc import (
	build_optimizer, build_scheduler, build_tensorboard_writer, build_checkpoint, get_prefix, main_run
)
from sslh.utils.recorder.recorder import Recorder
from sslh.utils.save import save_results
from sslh.utils.types import str_to_optional_str, str_to_bool
from sslh.validation.validater import Validater

from time import time
from torch.nn import MSELoss, BCELoss
from typing import Dict, Optional, List

RUN_NAME = "MixMatchExp"


def create_args() -> Namespace:
	parser = ArgumentParser()
	add_common_args(parser)

	group_mm = parser.add_argument_group(RUN_NAME)
	group_mm.add_argument("--lambda_u", type=float, default=1.0,
		help="MixMatch, FixMatch and ReMixMatch \"lambda_u\" hyperparameter. "
		"Coefficient of unsupervised loss component. (default: 1.0)")
	group_mm.add_argument("--batch_size_u", "--bsize_u", type=int, default=128,
		help="Batch size used for unsupervised loader. (default: 128)")

	group_mm.add_argument("--criterion_s", type=str, default="ce",
		choices=["mse", "ce", "kl", "js", "bce"],
		help="MixMatch supervised loss component. (default: \"ce\")")
	group_mm.add_argument("--criterion_u", type=str, default="ce",
		choices=["mse", "ce", "kl", "js", "bce"],
		help="MixMatch unsupervised loss component. (default: \"ce\")")

	group_mm.add_argument("--nb_augms", type=int, default=2,
		help="Nb of augmentations used in MixMatch. (default: 2)")
	group_mm.add_argument("--temperature", "--sharpen_temperature", type=float, default=0.5,
		help="MixMatch and ReMixMatch hyperparameter temperature used by sharpening. (default: 0.5)")
	group_mm.add_argument("--alpha", "--mixup_alpha", type=float, default=0.75,
		help="MixMatch and ReMixMatch hyperparameter \"alpha\" used by MixUp. (default: 0.75)")

	group_mm.add_argument("--warmup_nb_steps", type=int, default=16000,
		help="Nb of steps when lambda_u and lambda_u1 is increase from 0 to their value. "
		"Use 0 for deactivate warmup. (default: 16000)")

	group_mm.add_argument("--backward_frequency", type=int, default=1,
		help="Activate accumulative loss in order to update model not on every iteration. (default: 1)")

	group_mm.add_argument("--use_true_label_u", type=str_to_bool, default=False,
		help="Get the true label for compute metrics on unlabeled data. (default: False)")

	group_mm.add_argument("--use_adversarial", "--use_adv", type=str_to_bool, default=False,
		help="Use adversarial data instead of augmentations in MixMatch. (default: False)")
	group_mm.add_argument("--epsilon_adv", type=float, default=1e-2,
		help="Epsilon used in adversarial generator. (default: 1e-2)")
	group_mm.add_argument("--augm_none", type=str_to_optional_str, default=None,
		help="Augment pool for no-augmentation to use for adversarial training. (default: None)")

	group_mm.add_argument("--use_warmup_by_epoch", type=str_to_bool, default=False,
		help="Use increase lambda_u only at the end of an epoch. (default: False)")

	group_mm.add_argument("--use_no_label_mix", type=str_to_bool, default=False,
		help="Do not mix labels and use a special criterion MixMatchLossNoLabelMix similar to MixUpLoss. (default: False)")

	group_mm.add_argument("--augm_weak", type=str_to_optional_str, default="weak",
		help="Augment pool for weak augmentation to use. (default: \"weak\")")

	group_mm.add_argument("--use_argmax", type=str_to_bool, default=False,
		help="Use argmax instead of sharpen for post process guessed labels. (default: False)")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	return args


def run_mixmatch_exp(
	args: Namespace,
	start_date: str,
	interface: DatasetInterface,
	folds_train: Optional[List[int]],
	folds_val: Optional[List[int]],
	device: torch.device,
) -> Dict[str, Dict[str, float]]:
	"""
		Run a MixMatch Experimental training.

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
	transform_none = get_transform(args.augm_none, args, data_type, transform_base)
	transform_weak = get_transform(args.augm_weak, args, data_type, transform_base)
	transform_val = transform_base
	target_transform = interface.get_target_transform()

	dataset_train_raw = interface.get_dataset_train(args.dataset_path, folds=folds_train)
	dataset_val = interface.get_dataset_val(args.dataset_path, transform_val, target_transform, folds=folds_val)
	dataset_eval = interface.get_dataset_eval(args.dataset_path, transform_val, target_transform)

	def transform_weak_label(item: tuple) -> tuple:
		data, label = item
		return transform_weak(data), target_transform(label)

	def transform_weaks_no_label(item: tuple) -> tuple:
		data, label = item
		return tuple([transform_weak(data) for _ in range(args.nb_augms)])

	def transform_weaks(item: tuple) -> tuple:
		data, label = item
		return tuple([transform_weak(data) for _ in range(args.nb_augms)] + [target_transform(label)])

	def transform_none_label(item: tuple) -> tuple:
		data, label = item
		return transform_none(data), target_transform(label)

	if not args.use_adversarial:
		dataset_train_augm_weak = TransformDataset(
			dataset_train_raw, transform=transform_weak_label, index=None,
		)

		if not args.use_true_label_u:
			transform_augm = transform_weaks_no_label
		else:
			transform_augm = transform_weaks

		dataset_train_augm_weaks_no_label = TransformDataset(
			dataset_train_raw, transform=transform_augm, index=None,
		)

		loader_train_s, loader_train_u = interface.get_loaders_split(
			labeled_dataset=dataset_train_raw,
			ratios=[args.supervised_ratio, 1.0 - args.supervised_ratio],
			datasets=[dataset_train_augm_weak, dataset_train_augm_weaks_no_label],
			batch_sizes=[args.batch_size_s, args.batch_size_u],
			drop_last_list=[True, True],
			num_workers_list=[2, 6],
			target_transformed=False,
		)

	else:
		dataset_train_augm_none = TransformDataset(
			dataset_train_raw, transform=transform_none_label, index=None,
		)
		dataset_train_multiple_no_label = TransformDataset(
			dataset_train_raw, transform=transform_weaks_no_label, index=None,
		)

		loader_train_s, loader_train_u = interface.get_loaders_split(
			labeled_dataset=dataset_train_raw,
			ratios=[args.supervised_ratio, 1.0 - args.supervised_ratio],
			datasets=[dataset_train_augm_none, dataset_train_multiple_no_label],
			batch_sizes=[args.batch_size_s, args.batch_size_u],
			drop_last_list=[True, True],
			num_workers_list=[2, 6],
			target_transformed=False,
		)

	loader_train = ZipCycle([loader_train_s, loader_train_u], policy=args.zip_cycle_policy)
	loader_val = interface.get_loader_val(dataset_val, batch_size=args.batch_size_s, drop_last=False, num_workers=0)

	# Prepare model
	model = get_model(args.model, args, device=device)
	model_name = model.__class__.__name__
	optim = build_optimizer(args, model)
	activation = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)

	loss_mapper = {
		"mse": MSELoss(),
		"ce": CrossEntropyWithVectors(),
		"kl": KLDivLossWithProbabilities(),
		"js": JSDivLoss(),
		"bce": BCELoss(),
	}
	criterion_s = loss_mapper[args.criterion_s]
	criterion_u = loss_mapper[args.criterion_u]
	criterion = MixMatchLoss(criterion_s, criterion_u)

	sched = build_scheduler(args, optim)

	# Prepare metrics
	target_type = interface.get_target_type()
	if target_type == "monolabel":
		main_metric_name = "val/acc"

		metrics_train_s_mix = {"train/acc_s": CategoricalAccuracy(dim=1)}
		metrics_train_u_mix = {"train/acc_u": CategoricalAccuracy(dim=1)}
		metrics_val = {
			"val/acc": CategoricalAccuracy(dim=1),
			"val/ce": MetricWrapper(CrossEntropyWithVectors(dim=1)),
			"val/max": MetricWrapper(Max(dim=1), use_target=False, reduce_fn=torch.mean),
		}
	elif target_type == "multilabel":
		main_metric_name = "val/mAP"
		metrics_train_s_mix = {"train/mAP_s": AveragePrecision(), "train/mAUC_s": RocAuc(), "train/dPrime_s": DPrime()}
		metrics_train_u_mix = {"train/mAP_u": AveragePrecision(), "train/mAUC_u": RocAuc(), "train/dPrime_u": DPrime()}
		metrics_val = {"val/mAP": AveragePrecision(), "val/mAUC": RocAuc(), "val/dPrime": DPrime()}
	else:
		raise RuntimeError(f"Unknown target type \"{target_type}\".")

	# Prepare objects for saving data
	prefix = get_prefix(args, folds_val, interface, start_date, model_name, RUN_NAME)
	writer, dirpath_writer = build_tensorboard_writer(args, prefix)
	recorder = Recorder(writer)
	checkpoint = build_checkpoint(args, dirpath_writer, model, optim)

	# Start main training
	if args.backward_frequency > 1:
		trainer = MixMatchTrainerAcc(
			model, activation, optim, loader_train, metrics_train_s_mix, metrics_train_u_mix, recorder, criterion,
			device=device,
			temperature=args.temperature,
			alpha=args.alpha,
			lambda_u=args.lambda_u,
			warmup_nb_steps=args.warmup_nb_steps,
			backward_frequency=args.backward_frequency,
		)
	elif args.use_true_label_u:
		if target_type == "monolabel":
			metrics_s = {"acc_s": CategoricalAccuracy(dim=1)}
			metrics_u = {"acc_u": CategoricalAccuracy(dim=1)}
		else:
			metrics_s = {"train/mAP_s": AveragePrecision(), "train/mAUC_s": RocAuc(), "train/dPrime_s": DPrime()}
			metrics_u = {"train/mAP_u": AveragePrecision(), "train/mAUC_u": RocAuc(), "train/dPrime_u": DPrime()}

		trainer = MixMatchTrainerTrueLabel(
			model, activation, optim, loader_train, metrics_train_s_mix, metrics_train_u_mix, recorder, criterion,
			device=device,
			temperature=args.temperature,
			alpha=args.alpha,
			lambda_u=args.lambda_u,
			warmup_nb_steps=args.warmup_nb_steps,
			metrics_s=metrics_s,
			metrics_u=metrics_u,
		)
	elif args.use_adversarial:
		trainer = MixMatchTrainerAdv(
			model, activation, optim, loader_train, metrics_train_s_mix, metrics_train_u_mix, recorder, criterion,
			device=device,
			temperature=args.temperature,
			alpha=args.alpha,
			lambda_u=args.lambda_u,
			warmup_nb_steps=args.warmup_nb_steps,
			criterion_adv=CrossEntropyWithVectors(),
			epsilon_adv=args.epsilon_adv,
		)
	elif args.use_warmup_by_epoch:
		trainer = MixMatchTrainerNoWarmUp(
			model, activation, optim, loader_train, metrics_train_s_mix, metrics_train_u_mix, recorder, criterion,
			device=device,
			temperature=args.temperature,
			alpha=args.alpha,
			lambda_u=args.lambda_u,
		)
		warmup = WarmUp(max_value=args.lambda_u, nb_steps=args.warmup_nb_steps, obj=trainer, attr_name="lambda_u")
		trainer.add_callback_on_end(warmup)
	elif args.use_no_label_mix:
		criterion = MixMatchLossNoLabelMix()
		trainer = MixMatchTrainerNoLabelMix(
			model, activation, optim, loader_train, metrics_train_s_mix, metrics_train_u_mix, recorder, criterion,
			device=device,
			temperature=args.temperature,
			alpha=args.alpha,
			lambda_u=args.lambda_u,
			warmup_nb_steps=args.warmup_nb_steps,
		)
	elif args.use_argmax:
		trainer = MixMatchTrainerArgmax(
			model, activation, optim, loader_train, metrics_train_s_mix, metrics_train_u_mix, recorder, criterion,
			device=device,
			temperature=args.temperature,
			alpha=args.alpha,
			lambda_u=args.lambda_u,
			warmup_nb_steps=args.warmup_nb_steps,
		)
	else:
		trainer = MixMatchTrainer(
			model, activation, optim, loader_train, metrics_train_s_mix, metrics_train_u_mix, recorder, criterion,
			device=device,
			temperature=args.temperature,
			alpha=args.alpha,
			lambda_u=args.lambda_u,
			warmup_nb_steps=args.warmup_nb_steps,
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

	print("Dataset : {:s} (train={:d}, val={:d}, eval={:s}).".format(
		args.dataset_name,
		len(dataset_train_raw),
		len(dataset_val),
		str(len(dataset_eval)) if dataset_eval is not None else "None"
	))
	print("\nStart {:s} training on {:s} with model \"{:s}\" and {:d} epochs ({:s})...".format(RUN_NAME, args.dataset_name, model_name, args.nb_epochs, args.tag))
	start_time = time()

	for epoch in range(args.nb_epochs):
		trainer.train(epoch)
		validater.val(epoch)
		print()

	duration = time() - start_time
	print("\nEnd {:s} training. (duration = {:.2f})".format(RUN_NAME, duration))

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
	main_run(create_args, run_mixmatch_exp, RUN_NAME)


if __name__ == "__main__":
	main()
