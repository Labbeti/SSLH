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

from mlu.utils.misc import get_nb_parameters

from sslh.augments.get_pool import get_transform
from sslh.datasets.get_builder import DatasetBuilder
from sslh.models.checkpoint import load_state
from sslh.models.get_model import get_model
from sslh.utils.args import post_process_args, check_args, add_common_args
from sslh.utils.misc import main_run, get_activation
from sslh.validation.models_comparator import ModelsComparator

from time import time
from typing import Dict, List, Optional


RUN_NAME = "Comparison"


def create_args() -> Namespace:
	parser = ArgumentParser()
	parser = add_common_args(parser)

	group_comp = parser.add_argument_group(f"{RUN_NAME} args")

	group_comp.add_argument("--models", type=str, nargs="+", default=[], required=True,
		help="Path to the checkpoint model saved. (default: [])")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	for filepath_model in args.models:
		if not osp.isfile(filepath_model):
			raise RuntimeError("Invalid filepath '{:s}'.".format(filepath_model))

	return args


def run_compare(
	args: Namespace,
	start_date: str,
	git_hash: str,
	builder: DatasetBuilder,
	folds_train: Optional[List[int]],
	folds_val: Optional[List[int]],
	device: torch.device,
) -> Dict[str, Dict[str, float]]:
	"""
		Run a Comparison test.

		:param args: The argparse arguments fo the run.
		:param start_date: Date of the start of the run.
		:param git_hash: The current git hash of the repository.
		:param folds_train: The folds used for training the model.
		:param folds_val: The folds used for validating the model.
		:param builder: The dataset builder used for training.
		:param device: The main Pytorch device to use.
		:return: A dictionary containing the min and max scores on all epochs.
	"""
	transform_val = get_transform("identity", args, builder)
	target_transform = builder.get_target_transform()

	dataset_val = builder.get_dataset_val(args.dataset_path, transform_val, target_transform, folds=folds_val)
	loader_val = builder.get_loader_val(dataset_val, args.batch_size_s)

	# Prepare model
	activation = get_activation(args.activation, clamp=True, clamp_min=2e-30)

	models = []
	for filepath_model in args.models:
		model = get_model(args.model, args, builder, device)
		if args.nb_gpu > 1:
			model = DataParallel(model)
		load_state(filepath_model, model, None)
		models.append(model)

	print("Dataset : {:s} (val={:d}).".format(args.dataset_name, len(dataset_val)))
	print("Models: ")
	for model in models:
		print(f"{model.__class__.__name__} : {get_nb_trainable_parameters(model)}")

	print("\nStart {:s} training with {:d} epochs (tag: '{:s}')...".format(RUN_NAME, args.nb_epochs, args.tag))
	start_time = time()

	comparator = ModelsComparator(models, activation, loader_val, {})
	comparator.val(0)

	print("End {:s}. (duration = {:.2f})".format(RUN_NAME, time() - start_time))

	nb_models = len(args.models)
	matrix = comparator.get_matrix()
	total_occ = matrix.sum()
	model_nums = " ".join(["{:d}".format(i) for i in range(nb_models)])

	print("\nModels: ")
	for i, model_path in enumerate(args.models):
		print("{:2d}: {:s}".format(i, model_path))
	print()

	print("| {:s} | {:>6s} | {:>6s} | ".format(model_nums, "occ", "%"))
	for idx, occ in enumerate(matrix):
		bin_idx = str(bin(idx))[2:][::-1]
		bin_idx = bin_idx.replace("0", "F").replace("1", "T")
		bin_idx += "F" * (nb_models - len(bin_idx))
		correct_models = " ".join(bin_idx)
		print("| {:s} | {:6d} | {:>6.2f} |".format(correct_models, occ, occ / total_occ * 100.0))

	return {}


def main():
	main_run(create_args, run_compare, RUN_NAME)


if __name__ == "__main__":
	main()
