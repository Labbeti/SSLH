"""
	Main script for running a ReMixMatch training.
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

from ssl.datasets.get_interface import get_dataset_interface, DatasetInterface
from ssl.remixmatch.loss import ReMixMatchLoss
from ssl.remixmatch.trainer import ReMixMatchTrainer
from ssl.utils.args import post_process_args, check_args, add_common_args
from ssl.utils.cross_validation import cross_validation
from ssl.utils.misc import build_optimizer, build_scheduler, build_tensorboard_writer, build_checkpoint, get_prefix
from ssl.utils.other_metrics import CategoricalAccuracyOnehot, CrossEntropyMetric, EntropyMetric, MaxMetric
from ssl.utils.recorder.recorder import Recorder
from ssl.utils.save import save_results
from ssl.utils.types import str_to_optional_str
from ssl.validation.validater import Validater

from time import time
from torch.utils.data import DataLoader, Subset
from typing import Optional, Dict, Union


TRAIN_NAME = "ReMixMatch"


def create_args() -> Namespace:
	parser = ArgumentParser()
	add_common_args(parser)

	group_rmm = parser.add_argument_group(TRAIN_NAME)
	group_rmm.add_argument("--lambda_u", type=float, default=1.5,
		help="MixMatch, FixMatch and ReMixMatch \"lambda_u\" hyperparameter. "
			 "Coefficient of unsupervised loss component. (default: 1.5)")
	group_rmm.add_argument("--batch_size_u", "--bsize_u", type=int, default=128,
		help="Batch size used for unsupervised loader. (default: 128)")

	group_rmm.add_argument("--nb_augms", type=int, default=8,
		help="Nb of strong augmentations used in ReMixMatch. (default: 8)")
	group_rmm.add_argument("--temperature", "--sharpen_temperature", type=float, default=0.5,
		help="MixMatch and ReMixMatch hyperparameter temperature used by sharpening. (default: 0.5)")
	group_rmm.add_argument("--alpha", "--mixup_alpha", type=float, default=0.75,
		help="MixMatch and ReMixMatch hyperparameter \"alpha\" used by MixUp. (default: 0.75)")

	group_rmm.add_argument("--lambda_u1", type=float, default=0.5,
		help="ReMixMatch \"lambda_u1\" hyperparameter. Coefficient of direct unsupervised loss component. (default: 0.5)")
	group_rmm.add_argument("--lambda_r", type=float, default=0.5,
		help="ReMixMatch \"lambda_r\" hyperparameter. Coefficient of rotation loss component. (default: 0.5)")
	group_rmm.add_argument("--history", "--history_size", type=int, default=128,
		help="Nb of batch kept for compute prediction means in ReMixMatch. (default: 128)")

	group_rmm.add_argument("--augm_weak", type=str_to_optional_str, default="weak",
		help="Augment pool for weak augmentation to use. (default: weak)")
	group_rmm.add_argument("--augm_strong", type=str_to_optional_str, default="strong",
		help="Augment pool for strong augmentation to use. (default: strong)")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	if args.batch_size_s != args.batch_size_u:
		raise RuntimeError("Batch sizes must be equal for ReMixMatch.")

	return args


def run_remixmatch(args: Namespace, start_date: str, fold_val: Optional[int], interface: DatasetInterface) -> Dict[str, Union[float, int]]:
	# Build loaders
	if interface.get_nb_folds() is None:
		folds_train, folds_val = None, None
	else:
		if fold_val is None:
			fold_val = interface.get_nb_folds()
		folds_train = list(range(1, interface.get_nb_folds() + 1))
		folds_train.remove(fold_val)
		folds_val = [fold_val]

	dataset_train_augm_weak = interface.get_dataset_train_augm_weak(args, folds_train)
	dataset_train_augm_strong = interface.get_dataset_train_augm_strong(args, folds_train)
	dataset_val = interface.get_dataset_val(args, folds_val)
	dataset_eval = interface.get_dataset_eval(args, None)

	indexes_s, indexes_u = interface.generate_indexes_for_split(dataset_train_augm_weak, [args.supervised_ratio, 1.0 - args.supervised_ratio])
	dataset_train_s_augm_strong = Subset(dataset_train_augm_strong, indexes_s)
	dataset_train_u_augm_weak = Subset(dataset_train_augm_weak, indexes_u)
	dataset_train_u_augm_strong = Subset(dataset_train_augm_strong, indexes_u)

	dataset_train_u_augm_weak = NoLabelDataset(dataset_train_u_augm_weak)
	dataset_train_u_augm_strong = NoLabelDataset(dataset_train_u_augm_strong)

	dataset_train_u_augm_strong_multiple = ZipDataset([dataset_train_u_augm_strong] * args.nb_augms)
	dataset_train_u_augm_weak_strong = ZipDataset([dataset_train_u_augm_weak, dataset_train_u_augm_strong_multiple])

	loader_train_s_augm_strong = DataLoader(
		dataset=dataset_train_s_augm_strong, batch_size=args.batch_size_s, shuffle=True, num_workers=2, drop_last=True)
	loader_train_u_augm_weak_strong = DataLoader(
		dataset=dataset_train_u_augm_weak_strong, batch_size=args.batch_size_u, shuffle=True, num_workers=6,
		drop_last=True)

	loader_train = ZipCycle([loader_train_s_augm_strong, loader_train_u_augm_weak_strong])
	loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=False)

	# Prepare model
	model = interface.build_model(args.model, args)
	model_name = model.__class__.__name__
	optim = build_optimizer(args, model)
	activation = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)

	criterion = ReMixMatchLoss()

	sched = build_scheduler(args, optim)

	# Prepare metrics
	main_metric_name = "val/acc"
	metrics_train_s_mix = {"acc_s_mix": CategoricalAccuracyOnehot(dim=1)}
	metrics_train_u_mix = {"acc_u_mix": CategoricalAccuracyOnehot(dim=1)}
	metrics_train_u1 = {"acc_u1": CategoricalAccuracyOnehot(dim=1)}
	metrics_train_r = {"acc_r": CategoricalAccuracyOnehot(dim=1)}
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
	trainer = ReMixMatchTrainer(
		model, activation, optim, loader_train, metrics_train_s_mix, metrics_train_u_mix, metrics_train_u1, metrics_train_r, recorder,
		transform_self_supervised=interface.get_transform_self_supervised(args),
		criterion=criterion,
		temperature=args.temperature,
		alpha=args.alpha,
		lambda_u=args.lambda_u,
		lambda_u1=args.lambda_u1,
		lambda_r=args.lambda_r,
		history=args.history,
	)
	validator = Validater(model, activation, loader_val, metrics_val, recorder, checkpoint=checkpoint, checkpoint_metric=main_metric_name)

	if sched is not None:
		validator.add_callback_on_end(sched)

	print("Dataset : {:s} (train={:d} (supervised={:d}, unsupervised={:d}), val={:d}, eval={:s}).".format(
		args.dataset_name,
		len(dataset_train_augm_weak),
		len(dataset_train_s_augm_strong),
		len(dataset_train_u_augm_weak),
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
		cross_validation(run_remixmatch, args, start_date, interface, TRAIN_NAME)
	else:
		run_remixmatch(args, start_date, args.fold_val, interface)

	exec_time = time() - start_time
	print("")
	print("Program started at \"{:s}\" and terminated at \"{:s}\".".format(start_date, get_datetime()))
	print("Total execution time: {:.2f}s".format(exec_time))


if __name__ == "__main__":
	main()
