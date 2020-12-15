"""
	Main script for evaluate a model saved on evaluation dataset (available for GoogleSpeechCommand only).
"""

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import os.path as osp
import torch

from argparse import ArgumentParser, Namespace

from mlu.utils.misc import get_datetime, reset_seed

from sslh.datasets.get_interface import get_dataset_interface, DatasetInterface
from sslh.utils.args import post_process_args, check_args, add_common_args
from sslh.models.checkpoint import CheckPoint
from sslh.utils.other_metrics import CategoricalAccuracyOnehot, CrossEntropyMetric, EntropyMetric
from sslh.utils.recorder.recorder import Recorder
from sslh.utils.types import str_to_optional_str
from sslh.validation.validater import Validater

from time import time
from torch.utils.data import DataLoader
from typing import Optional, Dict, Union


TRAIN_NAME = "Evaluation"


def create_args() -> Namespace:
	parser = ArgumentParser()
	add_common_args(parser)

	parser.add_argument("--pre_trained_model", type=str_to_optional_str, default=None, required=True,
		help="Path to the checkpoint model saved. (default: None)")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	if not osp.isfile(args.pre_trained_model):
		raise RuntimeError("Invalid filepath \"{:s}\".".format(args.pre_trained_model))

	return args


def run_eval(args: Namespace, start_date: str, fold_val: Optional[int], interface: DatasetInterface) -> Dict[str, Union[float, int]]:
	# Build loaders
	dataset_eval = interface.get_dataset_eval(args)
	loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size_s, shuffle=False, drop_last=False)

	# Prepare model
	model = interface.build_model(args.model, args)
	model_name = model.__class__.__name__
	activation = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)

	CheckPoint.load_best_state(args.pre_trained_model, model, None)

	# Prepare metrics
	metrics_eval = {"acc": CategoricalAccuracyOnehot(dim=1), "ce": CrossEntropyMetric(dim=1), "entropy": EntropyMetric(dim=1)}
	recorder = Recorder()

	validator = Validater(model, activation, loader_eval, metrics_eval, recorder, name="eval")

	print("Dataset : {:s} (eval={:d}).".format(args.dataset_name, len(dataset_eval)))
	print("\nStart {:s} evaluation on {:s} with model \"{:s}\" and {:d} epochs ({:s})...".format(TRAIN_NAME, args.dataset_name, model_name, args.nb_epochs, args.tag))
	start_time = time()

	validator.val(0)

	print("End {:s}. (duration = {:.2f})".format(TRAIN_NAME, time() - start_time))

	main_metric_name = "eval/acc"
	best = recorder.get_best_epoch(main_metric_name)
	print("Metric : \"{:s}\"".format(main_metric_name))
	print("Best epoch : {:d}".format(best["best_epoch"]))
	print("Best mean : {:f}".format(best["best_mean"]))
	print("Best std : {:f}".format(best["best_std"]))

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
	run_eval(args, start_date, args.fold_val, interface)

	exec_time = time() - start_time
	print("")
	print("Program started at \"{:s}\" and terminated at \"{:s}\".".format(start_date, get_datetime()))
	print("Total execution time: {:.2f}s".format(exec_time))


if __name__ == "__main__":
	main()
