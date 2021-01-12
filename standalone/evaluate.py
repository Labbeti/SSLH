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

from mlu.metrics import CategoricalAccuracy, MetricWrapper
from mlu.nn import CrossEntropyWithVectors, Entropy, Max

from sslh.dataset.get_interface import DatasetInterface
from sslh.models.checkpoint import CheckPoint
from sslh.models.get_model import get_model
from sslh.utils.args import post_process_args, check_args, add_common_args
from sslh.utils.misc import main_run
from sslh.utils.recorder.recorder import Recorder
from sslh.utils.types import str_to_optional_str
from sslh.validation.validater import Validater

from time import time
from typing import Dict, List, Optional


RUN_NAME = "Evaluation"


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


def run_eval(
	args: Namespace,
	start_date: str,
	interface: DatasetInterface,
	folds_train: Optional[List[int]],
	folds_val: Optional[List[int]],
	device: torch.device,
) -> Dict[str, Dict[str, float]]:
	"""
		Run an Evaluation test.

		:param args: The argparse arguments fo the run.
		:param start_date: Date of the start of the run.
		:param folds_train: The folds used for training the model.
		:param folds_val: The folds used for validating the model.
		:param interface: The dataset interface used for training.
		:param device: The main Pytorch device to use.
		:return: A dictionary containing the min and max scores on all epochs.
	"""
	# Build loaders
	transform_base = interface.get_base_transform()
	target_transform = interface.get_target_transform()
	dataset_eval = interface.get_dataset_eval(args.dataset_path, transform_base, target_transform)
	loader_eval = interface.get_loader_val(dataset_eval, batch_size=args.batch_size_s, shuffle=False, drop_last=False)

	# Prepare model
	model = get_model(args.model, args, device=device)
	model_name = model.__class__.__name__
	activation = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)

	CheckPoint.load_best_state(args.pre_trained_model, model, None)

	# Prepare metrics
	metrics_eval = {
		"eval/acc": CategoricalAccuracy(dim=1),
		"eval/ce": MetricWrapper(CrossEntropyWithVectors(dim=1)),
		"eval/entropy": MetricWrapper(Entropy(dim=1), use_target=False),
		"eval/max": MetricWrapper(Max(dim=1), use_target=False),
	}
	recorder = Recorder()

	validater = Validater(model, activation, loader_eval, metrics_eval, recorder, device=device, name="eval")

	print("Dataset : {:s} (eval={:d}).".format(args.dataset_name, len(dataset_eval)))
	print("\nStart {:s} evaluation on {:s} with model \"{:s}\" and {:d} epochs ({:s})...".format(
		RUN_NAME, args.dataset_name, model_name, args.nb_epochs, args.tag))
	start_time = time()

	validater.val(0)

	print("End {:s}. (duration = {:.2f})".format(RUN_NAME, time() - start_time))

	main_metric_name = "eval/acc"
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
	main_run(create_args, run_eval, RUN_NAME)


if __name__ == "__main__":
	main()
