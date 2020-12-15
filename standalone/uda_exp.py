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

from mlu.datasets.wrappers import NoLabelDataset, ZipDataset
from mlu.utils.misc import get_datetime, reset_seed
from mlu.utils.zip_cycle import ZipCycle

from sslh.datasets.get_interface import get_dataset_interface, DatasetInterface
from sslh.uda.loss import UDALoss
from sslh.uda.trainer import UDATrainer
from sslh.uda.trainer_mixup import UDATrainerMixUp
from sslh.utils.args import post_process_args, check_args, add_common_args
from sslh.utils.cross_validation import cross_validation
from sslh.utils.misc import build_optimizer, build_scheduler, build_tensorboard_writer, build_checkpoint, get_prefix
from sslh.utils.other_metrics import CategoricalAccuracyOnehot, CrossEntropyMetric, EntropyMetric, MaxMetric
from sslh.utils.recorder.recorder import Recorder
from sslh.utils.save import save_results
from mlu.nn import CrossEntropyWithVectors, JSDivLoss, KLDivLossWithProbabilities
from sslh.utils.types import str_to_optional_str, str_to_bool
from sslh.validation.validater import Validater

from time import time
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Subset
from typing import Optional, Dict, Union


TRAIN_NAME = "UDAExp"


def create_args() -> Namespace:
	parser = ArgumentParser()
	add_common_args(parser)

	group_uda = parser.add_argument_group(TRAIN_NAME)
	group_uda.add_argument("--lambda_u", type=float, default=1.0,
		help="MixMatch, FixMatch and ReMixMatch \"lambda_u\" hyperparameter. "
			 "Coefficient of unsupervised loss component. (default: 1.0)")
	group_uda.add_argument("--batch_size_u", "--bsize_u", type=int, default=128,
		help="Batch size used for unsupervised loader. (default: 128)")

	group_uda.add_argument("--criterion_s", type=str, default="ce",
		choices=["mse", "ce", "kl", "js"],
		help="UDA supervised loss component. (default: ce)")
	group_uda.add_argument("--criterion_u", type=str, default="ce",
		choices=["mse", "ce", "kl", "js"],
		help="UDA unsupervised loss component. (default: ce)")

	group_uda.add_argument("--threshold", "--threshold_confidence", type=float, default=0.8,
		help="FixMatch threshold for compute confidence mask in loss. (default: 0.8)")
	group_uda.add_argument("--temperature", "--sharpen_temperature", type=float, default=0.4,
		help="MixMatch and ReMixMatch hyperparameter temperature used by sharpening. (default: 0.4)")
	group_uda.add_argument("--augment", "--augment_type", type=str_to_optional_str, default="strong",
		help="Augment type used in UDA training. (default: strong)")

	group_uda.add_argument("--use_mixup", type=str_to_bool, default=False,
		help="Apply MixUp between supervised and unsupervised data. (default: False)")
	group_uda.add_argument("--alpha", "--mixup_alpha", type=float, default=0.75,
		help="Alpha hyperparameter used in MixUp. (default: 0.75)")

	group_uda.add_argument("--augm_none", type=str_to_optional_str, default=None,
		help="Augment pool for default training dataset. (default: None)")
	group_uda.add_argument("--augm_strong", type=str_to_optional_str, default="strong",
		help="Augment pool for strong augmentation to use. (default: strong)")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	return args


def run_uda_exp(args: Namespace, start_date: str, fold_val: Optional[int], interface: DatasetInterface) -> Dict[str, Union[float, int]]:
	# Build loaders
	if interface.get_nb_folds() is None:
		folds_train, folds_val = None, None
	else:
		if fold_val is None:
			fold_val = interface.get_nb_folds()
		folds_train = list(range(1, interface.get_nb_folds() + 1))
		folds_train.remove(fold_val)
		folds_val = [fold_val]

	dataset_train = interface.get_dataset_train(args, folds_train)
	dataset_train_augm = interface.get_dataset_train_augm_strong(args, folds_train)
	dataset_val = interface.get_dataset_val(args, folds_val)
	dataset_eval = interface.get_dataset_eval(args, None)

	indexes_s, indexes_u = interface.generate_indexes_for_split(dataset_train, [args.supervised_ratio, 1.0 - args.supervised_ratio])
	dataset_train_s = Subset(dataset_train, indexes_s)
	dataset_train_u = Subset(dataset_train, indexes_u)
	dataset_train_u_augm_strong = Subset(dataset_train_augm, indexes_u)

	dataset_train_u = NoLabelDataset(dataset_train_u)
	dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

	dataset_train_u_augm_none_strong = ZipDataset([dataset_train_u, dataset_train_u_augm_strong])

	loader_train_s = DataLoader(
		dataset=dataset_train_s, batch_size=args.batch_size_s, shuffle=True, num_workers=2, drop_last=True)
	loader_train_u_none_augm = DataLoader(
		dataset=dataset_train_u_augm_none_strong, batch_size=args.batch_size_u, shuffle=True, num_workers=6, drop_last=True)

	loader_train = ZipCycle([loader_train_s, loader_train_u_none_augm])
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
	}
	criterion_s = loss_mapper[args.criterion_s]
	criterion_u = loss_mapper[args.criterion_u]
	criterion = UDALoss(criterion_s, criterion_u)

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
			threshold=args.threshold,
			lambda_u=args.lambda_u,
			temperature=args.temperature,
		)
	validator = Validater(model, activation, loader_val, metrics_val, recorder, checkpoint=checkpoint, checkpoint_metric=main_metric_name)

	if sched is not None:
		validator.add_callback_on_end(sched)

	print("Dataset : {:s} (train={:d} (supervised={:d}, unsupervised={:d}), val={:d}, eval={:s}).".format(
		args.dataset_name,
		len(dataset_train),
		len(dataset_train_s),
		len(dataset_train_u),
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
		cross_validation(run_uda_exp, args, start_date, interface, TRAIN_NAME)
	else:
		run_uda_exp(args, start_date, args.fold_val, interface)

	exec_time = time() - start_time
	print("")
	print("Program started at \"{:s}\" and terminated at \"{:s}\".".format(start_date, get_datetime()))
	print("Total execution time: {:.2f}s".format(exec_time))


if __name__ == "__main__":
	main()
