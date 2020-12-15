
import csv
import json
import os.path as osp

from argparse import Namespace
from ssl.datasets.abc import DatasetInterface
from typing import Callable, Dict, List, Union


def cross_validation(
	run: Callable,
	args: Namespace,
	start_date: str,
	interface: DatasetInterface,
	train_name: str,
):
	"""
		Run a cross-validation with a run method.
	"""
	if interface.get_nb_folds() is None:
		raise RuntimeError("Cross validation is not supported for the \"{:s}\" dataset.".format(interface.get_dataset_name()))

	cross_validation_results = {}
	for fold_val in range(1, interface.get_nb_folds() + 1):
		best = run(args, start_date, fold_val, interface)
		cross_validation_results[fold_val] = best

	print_cv_results(cross_validation_results)

	if args.write_results:
		supervised_ratio_suffix = "{:3.2f}%".format(args.supervised_ratio * 100.0)
		filename = "{:s}_{:s}_{:s}_{:s}_{:s}_cross_val_results".format(
			interface.get_dataset_name(), start_date, train_name, supervised_ratio_suffix, args.tag)

		save_cv_results(args.logdir, filename, cross_validation_results)


def print_cv_results(cross_validation_results: Dict[int, Dict[str, Union[float, int]]]):
	global_mean, global_std = _get_global_mean_std(cross_validation_results)
	csv_content = _get_csv_content(cross_validation_results)

	print("\n")
	print("Cross-validation global mean : ", global_mean)
	print("Cross-validation global std : ", global_std)

	print("Cross-validation results : ")
	for line in csv_content:
		print(",\t".join([str(elt) for elt in line]))


def save_cv_results(dirpath: str, filename: str, cross_validation_results: Dict[int, Dict[str, Union[float, int]]]):
	# Print cross-val results
	global_mean, global_std = _get_global_mean_std(cross_validation_results)

	# Save cross-val results
	filepath_json = osp.join(dirpath, "{:s}.json".format(filename))
	json_content = {"results": cross_validation_results, "global_mean": global_mean, "global_std": global_std}
	with open(filepath_json, "w") as file:
		json.dump(json_content, file, indent="\t")

	filepath_csv = osp.join(dirpath, "{:s}.csv".format(filename))
	with open(filepath_csv, "w") as file:
		writer = csv.writer(file)
		csv_content = _get_csv_content(cross_validation_results)
		writer.writerows(csv_content)


def _get_global_mean_std(cross_validation_results: Dict[int, Dict[str, Union[float, int]]]) -> (float, float):
	global_mean, global_std = 0.0, 0.0
	for fold, best in cross_validation_results.items():
		global_mean += best["best_mean"]
		global_std += best["best_std"]
	global_mean /= len(cross_validation_results)
	global_std /= len(cross_validation_results)
	return global_mean, global_std


def _get_csv_content(
	cross_validation_results: Dict[int, Dict[str, Union[float, int]]]
) -> List[List[Union[float, int]]]:
	global_mean, global_std = _get_global_mean_std(cross_validation_results)
	csv_content = [
		[fold for fold in cross_validation_results.keys()] + ["global"],
		[best["best_mean"] for best in cross_validation_results.values()] + [global_mean],
		[best["best_std"] for best in cross_validation_results.values()] + [global_std],
	]
	return csv_content
