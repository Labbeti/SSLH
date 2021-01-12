
import json
import os.path as osp
import torch

from argparse import Namespace
from sslh.dataset.abc import DatasetInterface
from typing import Callable, Dict, Union


def cross_validation(
	run: Callable,
	args: Namespace,
	start_date: str,
	interface: DatasetInterface,
	run_name: str,
	device: torch.device,
):
	"""
		Run a cross-validation with a run method.

		:param run: The run function to call.
		:param args: The argparse arguments.
		:param start_date: Date of the start of the training.
		:param interface: The dataset interface used.
		:param run_name: The name of the run.
		:param device: The Pytorch main device to use for training.
	"""
	if not interface.has_folds():
		raise RuntimeError("Cross validation is not supported for the \"{:s}\" dataset.".format(interface.get_dataset_name()))

	cross_validation_results = {}
	for fold_val in range(1, len(interface.get_folds()) + 1):
		folds_val = [fold_val]
		folds_train = set(interface.get_folds()).difference(folds_val)

		best = run(args, start_date, interface, folds_train, folds_val, device)
		cross_validation_results[fold_val] = best

	print_cv_results(cross_validation_results)

	if args.write_results:
		dataset_name = interface.get_dataset_name()
		supervised_ratio_suffix = "{:3.2f}%".format(args.supervised_ratio * 100.0)
		filename = \
			f"{dataset_name}_{start_date}_{run_name}_{supervised_ratio_suffix}_{args.tag}_cross_val_results"

		save_cv_results(args.logdir, filename, cross_validation_results)


def print_cv_results(cross_validation_results: Dict[int, Dict[str, Dict[str, Union[float, int]]]]):
	global_scores = _get_global_scores(cross_validation_results)

	print("\n")
	print("Cross-validation results : ")
	for metric_name, scores in global_scores.items():
		print(f"- Metric \"{metric_name}\" :")

		for fold, bests in cross_validation_results.items():
			min_, max_ = bests[metric_name]["min"], bests[metric_name]["max"]
			print(f"\t- Fold {fold} : [min: {min_}, \tmax: {max_}]")

		mean_min, mean_max, std_min, std_max = scores["mean_min"], scores["mean_max"], scores["std_min"], scores["std_max"]
		print(f"\t- Global mean mins : {mean_min} +/- {std_min}")
		print(f"\t- Global mean maxs : {mean_max} +/- {std_max}")


def save_cv_results(
	dirpath: str, filename: str, cross_validation_results: Dict[int, Dict[str, Dict[str, Union[float, int]]]]
):
	# Print cross-val results
	global_scores = _get_global_scores(cross_validation_results)

	# Save cross-val results
	filepath_json = osp.join(dirpath, f"{filename}.json")
	json_content = {
		"results_by_folds": cross_validation_results,
		"global_scores": global_scores,
	}

	with open(filepath_json, "w") as file:
		json.dump(json_content, file, indent="\t")


def _get_global_scores(
	cross_validation_results: Dict[int, Dict[str, Dict[str, Union[float, int]]]]
) -> Dict[str, Dict[str, float]]:
	nb_folds = len(cross_validation_results)

	global_scores = {}
	for metric_name in cross_validation_results[1].keys():
		mins = [bests[metric_name]["min"] for bests in cross_validation_results.values()]
		maxs = [bests[metric_name]["max"] for bests in cross_validation_results.values()]
		global_scores[metric_name] = {
			"mean_min": torch.as_tensor(mins).mean().item(),
			"mean_max": torch.as_tensor(maxs).mean().item(),
			"std_min": torch.as_tensor(mins).std(unbiased=False).item(),
			"std_max": torch.as_tensor(maxs).std(unbiased=False).item(),
		}
	return global_scores
