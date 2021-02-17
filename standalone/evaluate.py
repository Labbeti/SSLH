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
from mlu.nn import CrossEntropyWithVectors, Max
from mlu.utils.misc import get_nb_parameters

from sslh.augments.get_pool import get_transform
from sslh.datasets.get_builder import DatasetBuilder
from sslh.models.checkpoint import CheckPoint
from sslh.models.get_model import get_model
from sslh.utils.args import post_process_args, check_args, add_common_args
from sslh.utils.misc import main_run, get_activation
from sslh.utils.recorder.recorder import Recorder
from sslh.utils.types import str_to_optional_str
from sslh.validation.validater import Validater

from time import time
from typing import Dict, List, Optional


RUN_NAME = "Evaluation"


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser = add_common_args(parser)

	group_eval = parser.add_argument_group(f"{RUN_NAME} args")

	group_eval.add_argument("--pre_trained_model", type=str_to_optional_str, default=None,
		help="Path to the checkpoint model saved. (default: None)")

	group_eval.add_argument("--logdir", type=str_to_optional_str, default=None,
		help="Path to the logdir of a tensorboard where torch models where saved. (default: None)")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	return args


def run_eval(
	args: Namespace,
	start_date: str,
	builder: DatasetBuilder,
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
		:param builder: The dataset builder used for training.
		:param device: The main Pytorch device to use.
		:return: A dictionary containing the min and max scores on all epochs.
	"""
	# Build loaders
	transform_val = get_transform("identity", args, builder)
	target_transform = builder.get_target_transform()

	dataset_val = builder.get_dataset_val(args.dataset_path, transform_val, target_transform, folds=folds_val)
	loader_val = builder.get_loader_val(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=False)

	dataset_eval = builder.get_dataset_eval(args.dataset_path, transform_val, target_transform)
	loader_eval = builder.get_loader_val(dataset_eval, batch_size=args.batch_size_s, shuffle=False, drop_last=False)

	# Prepare model
	model = get_model(args.model, args, builder, device)
	if args.nb_gpu > 1:
		model = DataParallel(model)
	activation = get_activation(args.activation, clamp=True, clamp_min=2e-30)

	if args.pre_trained_model is not None:
		if not osp.isfile(args.pre_trained_model):
			raise RuntimeError(f"Invalid filepath '{args.pre_trained_model}'.")
		model_filepath = args.pre_trained_model
	elif args.logdir is not None:
		if not osp.isdir(args.logdir):
			raise RuntimeError(f"Invalid dirpath '{args.logdir}'.")
		objects_names = os.listdir(args.logdir)
		model_filepath = None
		for name in objects_names:
			path = osp.join(args.logdir, name)
			if osp.isfile(path) and path.endswith(".torch") and "rank_0" in path:
				model_filepath = path
				break
		if model_filepath is None:
			raise RuntimeError(f"Cannot find a '.torch' model of rank 0 in logdir '{args.logdir}'.")
	else:
		raise RuntimeError("No pre trained path or logdir path given.")
	CheckPoint.load_best_state(args.pre_trained_model, model, None)

	# Prepare metrics
	if builder.get_target_type() == "monolabel":
		metrics_val = {
			"val/acc": CategoricalAccuracy(dim=1),
			"val/ce": MetricWrapper(CrossEntropyWithVectors(dim=1)),
			"val/max": MetricWrapper(Max(dim=1), use_target=False),
		}
		metrics_eval = {
			"eval/acc": CategoricalAccuracy(dim=1),
			"eval/ce": MetricWrapper(CrossEntropyWithVectors(dim=1)),
			"eval/max": MetricWrapper(Max(dim=1), use_target=False),
		}
	else:
		raise NotImplementedError("TODO")

	recorder = Recorder()
	validator = Validater(model, activation, loader_val, metrics_val, recorder, device=device, name="val")
	validator.add_callback_on_end(recorder)

	evaluator = Validater(model, activation, loader_eval, metrics_eval, recorder, device=device, name="eval")
	evaluator.add_callback_on_end(recorder)

	print("Dataset : {:s} (val={:d}, eval={:d}).".format(args.dataset_name, len(dataset_val), len(dataset_eval)))
	print("Model: {:s} ({:d} parameters).".format(args.model, get_nb_parameters(model)))
	print("\nStart {:s} training with {:d} epochs (tag: '{:s}')...".format(RUN_NAME, args.nb_epochs, args.tag))
	start_time = time()

	validator.val(0)
	evaluator.val(0)

	print("End {:s}. (duration = {:.2f})".format(RUN_NAME, time() - start_time))

	return recorder.get_all_min_max()


def main():
	main_run(create_args, run_eval, RUN_NAME)


if __name__ == "__main__":
	main()
