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

from mlu.datasets.wrappers import NoLabelDataset, OnlyLabelDataset, ZipDataset
from mlu.nn import CrossEntropyWithVectors, JSDivLoss, KLDivLossWithProbabilities
from mlu.utils.misc import get_datetime, reset_seed
from mlu.utils.zip_cycle import ZipCycle

from ssl.datasets.get_interface import get_dataset_interface, DatasetInterface
from ssl.fixmatch.loss import FixMatchLoss
from ssl.fixmatch.trainer import FixMatchTrainer
from ssl.fixmatch.trainer_adv import FixMatchTrainerAdv
from ssl.fixmatch.trainer_mixup import FixMatchTrainerMixUp
from ssl.fixmatch.trainer_mixup_shuffle import FixMatchTrainerMixUpShuffle
from ssl.fixmatch.trainer_mixup_teacher import FixMatchTrainerMixUpTeacher
from ssl.fixmatch.trainer_teacher import FixMatchTrainerTeacher
from ssl.fixmatch.trainer_teacher_label_u import FixMatchTrainerTeacherLabelU
from ssl.fixmatch.trainer_uniloss import FixMatchTrainerUniLoss
from ssl.mixmatch.warmup import WarmUp
from ssl.utils.args import post_process_args, check_args, add_common_args
from ssl.utils.cross_validation import cross_validation
from ssl.utils.misc import build_optimizer, build_scheduler, build_tensorboard_writer, build_checkpoint, get_prefix
from ssl.utils.other_metrics import CategoricalAccuracyOnehot, CrossEntropyMetric, EntropyMetric, MaxMetric
from ssl.utils.recorder.recorder import Recorder
from ssl.utils.save import save_results
from ssl.utils.types import str_to_bool, str_to_optional_str
from ssl.validation.validater import Validater

from time import time
from torch.nn import MSELoss, BCELoss
from torch.utils.data import DataLoader, Subset
from typing import Optional, Dict, Union


TRAIN_NAME = "FixMatchExp"


def create_args() -> Namespace:
	parser = ArgumentParser()
	add_common_args(parser)

	group_fm = parser.add_argument_group(TRAIN_NAME)
	group_fm.add_argument("--lambda_u", type=float, default=1.0,
		help="MixMatch, FixMatch and ReMixMatch \"lambda_u\" hyperparameter. "
			 "Coefficient of unsupervised loss component. (default: 1.0)")
	group_fm.add_argument("--batch_size_u", "--bsize_u", type=int, default=128,
		help="Batch size used for unsupervised loader. (default: 128)")

	group_fm.add_argument("--criterion_s", type=str, default="ce",
		choices=["mse", "ce", "kl", "js", "bce"],
		help="FixMatch supervised loss component. (default: ce)")
	group_fm.add_argument("--criterion_u", type=str, default="ce",
		choices=["mse", "ce", "kl", "js", "bce"],
		help="FixMatch unsupervised loss component. (default: ce)")

	group_fm.add_argument("--threshold", "--threshold_confidence", type=float, default=0.95,
		help="FixMatch threshold for compute confidence mask in loss. (default: 0.95)")

	group_fm.add_argument("--use_mixup", type=str_to_bool, default=False,
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
		help="Augment pool for weak augmentation to use. (default: weak)")
	group_fm.add_argument("--augm_strong", type=str_to_optional_str, default="strong",
		help="Augment pool for strong augmentation to use. (default: strong)")

	group_fm.add_argument("--use_mixup_shuffle", type=str_to_bool, default=False,
		help="Use MixUp between batch with itself shuffled (no mix between labeled and unlabeled data). (default: False)")

	group_fm.add_argument("--use_teacher", type=str_to_bool, default=False,
		help="Use a teacher for guessing labels. The teacher is updated with an EMA.")
	group_fm.add_argument("--ema_decay", "--decay", type=float, default=0.999,
		help="EMA decay used by FixMatch-Teacher.")
	group_fm.add_argument("--use_teacher_true_label_u", type=str_to_bool, default=False,
		help="Use a teacher for guessing labels. The teacher is updated with an EMA.")

	group_fm.add_argument("--use_warmup", type=str_to_bool, default=False,
		help="Use WarmUp on lambda_u hparam every epochs. Compatible with most of the training variants.")
	group_fm.add_argument("--warmup_nb_epochs", type=int, default=10,
		help="The number of epochs before warmup reach the lambda_u max value.")

	group_fm.add_argument("--use_mixup_teacher", type=str_to_bool, default=False,
		help="Combine MixUp and Teacher with FixMatch.")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	return args


def run_fixmatch_exp(args: Namespace, start_date: str, fold_val: Optional[int], interface: DatasetInterface) -> Dict[str, Union[float, int]]:
	# Build loaders
	if interface.get_nb_folds() is None:
		folds_train, folds_val = None, None
	else:
		if fold_val is None:
			fold_val = interface.get_nb_folds()
		folds_train = list(range(1, interface.get_nb_folds() + 1))
		folds_train.remove(fold_val)
		folds_val = [fold_val]

	if not args.use_adversarial:
		dataset_train_augm_weak = interface.get_dataset_train_augm_weak(args, folds_train)
		dataset_train_augm_strong = interface.get_dataset_train_augm_strong(args, folds_train)
		dataset_train = dataset_train_augm_weak

		indexes_s, indexes_u = interface.generate_indexes_for_split(dataset_train_augm_weak, [args.supervised_ratio, 1.0 - args.supervised_ratio])
		dataset_train_s_augm_weak = Subset(dataset_train_augm_weak, indexes_s)
		dataset_train_u_augm_weak = Subset(dataset_train_augm_weak, indexes_u)
		dataset_train_u_augm_strong = Subset(dataset_train_augm_strong, indexes_u)

		dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)
		dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

		if not args.use_teacher_true_label_u:
			dataset_train_u_augm_weak_strong = ZipDataset([dataset_train_u_augm_weak, dataset_train_u_augm_strong])
		else:
			dataset_train = interface.get_dataset_train(args, folds_train)
			dataset_train_u = Subset(dataset_train, indexes_u)
			dataset_train_u_labels = OnlyLabelDataset(dataset_train_u)

			dataset_train_u_augm_weak_strong = ZipDataset([dataset_train_u_augm_weak, dataset_train_u_augm_strong, dataset_train_u_labels])
	else:
		dataset_train = interface.get_dataset_train(args, folds_train)
		indexes_s, indexes_u = interface.generate_indexes_for_split(dataset_train, [args.supervised_ratio, 1.0 - args.supervised_ratio])

		dataset_train_s = Subset(dataset_train, indexes_s)
		dataset_train_u = Subset(dataset_train, indexes_u)
		dataset_train_u = NoLabelDataset(dataset_train_u)

		dataset_train_s_augm_weak = dataset_train_s
		dataset_train_u_augm_weak_strong = dataset_train_u

	dataset_val = interface.get_dataset_val(args, folds_val)
	dataset_eval = interface.get_dataset_eval(args, None)

	loader_train_s_augm = DataLoader(
		dataset_train_s_augm_weak, batch_size=args.batch_size_s, shuffle=True, num_workers=2, drop_last=True)
	loader_train_u_augms = DataLoader(
		dataset_train_u_augm_weak_strong, batch_size=args.batch_size_u, shuffle=True, num_workers=6, drop_last=True)

	loader_train = ZipCycle([loader_train_s_augm, loader_train_u_augms])
	loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=False)

	# Prepare model
	model = interface.build_model(args.model, args)
	model_name = model.__class__.__name__
	optim = build_optimizer(args, model)
	activation = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)

	loss_mapper = {
		"mse": MSELoss(reduction="none"),
		"ce": CrossEntropyWithVectors(reduction="none"),
		"kl": KLDivLossWithProbabilities(reduction="none"),
		"js": JSDivLoss(reduction="none"),
		"bce": BCELoss(reduction="none"),
	}
	criterion_s = loss_mapper[args.criterion_s]
	criterion_u = loss_mapper[args.criterion_u]
	criterion = FixMatchLoss(criterion_s, criterion_u)

	sched = build_scheduler(args, optim)

	# Prepare metrics
	main_metric_name = "val/acc"
	metrics_train_s = {"acc_s": CategoricalAccuracyOnehot(dim=1)}
	metrics_train_u = {"acc_u": CategoricalAccuracyOnehot(dim=1)}
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
	if args.use_mixup:
		trainer = FixMatchTrainerMixUp(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			alpha=args.alpha,
		)
	elif args.use_uniloss:
		trainer = FixMatchTrainerUniLoss(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder,
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
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			criterion_adv=CrossEntropyWithVectors(),
			epsilon_adv_weak=args.epsilon_adv_weak,
			epsilon_adv_strong=args.epsilon_adv_strong,
		)
	elif args.use_mixup_shuffle:
		trainer = FixMatchTrainerMixUpShuffle(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			alpha=args.alpha,
		)
	elif args.use_teacher:
		trainer = FixMatchTrainerTeacher(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			decay=args.ema_decay,
		)
	elif args.use_teacher_true_label_u:
		trainer = FixMatchTrainerTeacherLabelU(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			decay=args.ema_decay,
		)
	elif args.use_mixup_teacher:
		trainer = FixMatchTrainerMixUpTeacher(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			alpha=args.alpha,
			decay=args.ema_decay,
		)
	else:
		trainer = FixMatchTrainer(
			model, activation, optim, loader_train, metrics_train_s, metrics_train_u, recorder, criterion,
			threshold=args.threshold,
			lambda_u=args.lambda_u,
		)
	validator = Validater(model, activation, loader_val, metrics_val, recorder, checkpoint=checkpoint, checkpoint_metric=main_metric_name)

	if sched is not None:
		validator.add_callback_on_end(sched)

	if args.use_warmup:
		warmup = WarmUp(max_value=args.lambda_u, nb_steps=args.warmup_nb_epochs, obj=trainer, attr_name="lambda_u")
		validator.add_callback_on_end(warmup)

	print("Dataset : {:s} (train={:d} (supervised={:d}, unsupervised={:d}), val={:d}, eval={:s}).".format(
		args.dataset_name,
		len(dataset_train),
		len(dataset_train_s_augm_weak),
		len(dataset_train_u_augm_weak_strong),
		len(dataset_val),
		str(len(dataset_eval)) if dataset_eval is not None else "None"
	))
	print("\nStart {:s} training on {:s} with model \"{:s}\" and {:d} epochs ({:s})...".format(TRAIN_NAME, args.dataset_name, model_name, args.nb_epochs, args.tag))
	start_time = time()

	for epoch in range(args.nb_epochs):
		trainer.train(epoch)
		validator.val(epoch)
		print()

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
		cross_validation(run_fixmatch_exp, args, start_date, interface, TRAIN_NAME)
	else:
		run_fixmatch_exp(args, start_date, args.fold_val, interface)

	exec_time = time() - start_time
	print("")
	print("Program started at \"{:s}\" and terminated at \"{:s}\".".format(start_date, get_datetime()))
	print("Total execution time: {:.2f}s".format(exec_time))


if __name__ == "__main__":
	main()
