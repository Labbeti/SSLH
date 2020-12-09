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

from sslh.datasets.get_interface import get_dataset_interface, DatasetInterface
from sslh.utils.args import post_process_args, check_args, add_common_args
from sslh.models.checkpoint import CheckPoint
from sslh.utils.misc import get_datetime, reset_seed
from sslh.validation.comparator import Comparator

from time import time
from torch.utils.data import DataLoader
from typing import Optional, Dict, Union


TRAIN_NAME = "Comparison"


def create_args() -> Namespace:
	parser = ArgumentParser()
	add_common_args(parser)

	parser.add_argument("--models", type=str, nargs="+", default=[], required=True,
		help="Path to the checkpoint model saved. (default: [])")

	args = parser.parse_args()
	args = post_process_args(args)
	check_args(args)

	for filepath_model in args.models:
		if not osp.isfile(filepath_model):
			raise RuntimeError("Invalid filepath \"{:s}\".".format(filepath_model))

	return args


def run_compare(args: Namespace, start_date: str, fold_val: Optional[int], interface: DatasetInterface) -> Dict[str, Union[float, int]]:
	# Build loaders
	dataset_val = interface.get_dataset_val(args)
	loader_val = DataLoader(dataset_val, batch_size=args.batch_size_s, shuffle=False, drop_last=False)

	# Prepare model
	activation = lambda x, dim: x.softmax(dim=dim).clamp(min=2e-30)

	models = []
	for filepath_model in args.models:
		model = interface.build_model(args.model, args)
		CheckPoint.load_best_state(filepath_model, model, None)
		models.append(model)
	model_name = models[0].__class__.__name__

	print("Dataset : {:s} (val={:d}).".format(args.dataset_name, len(dataset_val)))
	print("\nStart {:s} evaluation on {:s} with model \"{:s}\" and {:d} epochs ({:s})...".format(TRAIN_NAME, args.dataset_name, model_name, args.nb_epochs, args.tag))
	start_time = time()

	comparator = Comparator(models, activation, loader_val)
	comparator.val(0)

	print("End {:s}. (duration = {:.2f})".format(TRAIN_NAME, time() - start_time))

	nb_models = len(args.models)
	matrix = comparator.get_matrix()
	total_occ = matrix.sum()
	model_nums = " ".join(["{:d}".format(i) for i in range(nb_models)])

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
	run_compare(args, start_date, args.fold_val, interface)

	exec_time = time() - start_time
	print("")
	print("Program started at \"{:s}\" and terminated at \"{:s}\".".format(start_date, get_datetime()))
	print("Total execution time: {:.2f}s".format(exec_time))


if __name__ == "__main__":
	main()
