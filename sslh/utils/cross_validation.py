
import json
import os.path as osp
import torch

from argparse import Namespace
from sslh.datasets.base import DatasetBuilder
from typing import Callable, Dict, Union


def cross_validation(
	run: Callable,
	args: Namespace,
	start_date: str,
	git_hash: str,
	builder: DatasetBuilder,
	run_name: str,
	device: torch.device,
):
	"""
		Run a cross-validation with a run method.

		:param run: The run function to call.
		:param args: The argparse arguments.
		:param start_date: Date of the start of the training.
		:param git_hash: The current git hash of the repository.
		:param builder: The dataset builder used.
		:param run_name: The name of the run.
		:param device: The Pytorch main device to use for training.
	"""
	if not builder.has_folds():
		raise RuntimeError("Cross validation is not supported for the '{:s}' dataset.".format(builder.get_dataset_name()))

	results_by_folds = {}
	for fold_val in range(1, len(builder.get_folds()) + 1):
		folds_val = [fold_val]
		folds_train = list(set(builder.get_folds()).difference(folds_val))

		best = run(args, start_date, git_hash, builder, folds_train, folds_val, device)
		results_by_folds[fold_val] = best

	print_cv_results(results_by_folds)

	if args.write_results:
		dataset_name = builder.get_dataset_name()
		supervised_ratio_suffix = "{:3.2f}%".format(args.supervised_ratio * 100.0)
		filename = \
			f"{dataset_name}_{start_date}_{run_name}_{supervised_ratio_suffix}_{args.tag}_cross_val_results"

		save_cv_results(args.logdir, filename, results_by_folds, args, start_date, git_hash, builder, run_name)


def print_cv_results(results_by_folds: Dict[int, Dict[str, Dict[str, Union[float, int]]]]):
	global_scores = _get_global_scores(results_by_folds)

	print("\n")
	print("Cross-validation results : ")
	for metric_name, scores in global_scores.items():
		print(f"- Metric '{metric_name}' :")

		for fold, bests in results_by_folds.items():
			min_, max_ = bests[metric_name]["min"], bests[metric_name]["max"]
			print(f"\t- Fold {fold} : [min: {min_}, \tmax: {max_}]")

		mean_min, mean_max, std_min, std_max = scores["mean_min"], scores["mean_max"], scores["std_min"], scores["std_max"]
		print(f"\t- Global mean mins : {mean_min} +/- {std_min}")
		print(f"\t- Global mean maxs : {mean_max} +/- {std_max}")


def save_cv_results(
	dirpath: str,
	filename: str,
	results_by_folds: Dict[int, Dict[str, Dict[str, Union[float, int]]]],
	args: Namespace,
	start_date: str,
	git_hash: str,
	builder: DatasetBuilder,
	run_name: str,
):
	# Print cross-val results
	global_scores = _get_global_scores(results_by_folds)

	info = {
		"run_name": run_name,
		"start_date": start_date,
		"tag": args.tag,
		"dataset": builder.get_dataset_name(),
		"model": args.model,
		"seed": args.seed,
		"supervised_ratio": args.supervised_ratio,
		"nb_epochs": args.nb_epochs,
		"git_hash": git_hash,
	}

	# Save cross-val results
	filepath_json = osp.join(dirpath, f"{filename}.json")
	json_content = {
		"results_by_folds": results_by_folds,
		"global_scores": global_scores,
		"info": info,
	}

	with open(filepath_json, "w") as file:
		json.dump(json_content, file, indent="\t")


def _get_global_scores(
	results_by_folds: Dict[int, Dict[str, Dict[str, Union[float, int]]]]
) -> Dict[str, Dict[str, float]]:
	global_scores = {}
	first_fold = list(results_by_folds.keys())[0]
	for metric_name in results_by_folds[first_fold].keys():
		mins = [bests[metric_name]["min"] for bests in results_by_folds.values()]
		maxs = [bests[metric_name]["max"] for bests in results_by_folds.values()]
		global_scores[metric_name] = {
			"mean_min": torch.as_tensor(mins).mean().item(),
			"mean_max": torch.as_tensor(maxs).mean().item(),
			"std_min": torch.as_tensor(mins).std(unbiased=False).item(),
			"std_max": torch.as_tensor(maxs).std(unbiased=False).item(),
		}
	return global_scores
