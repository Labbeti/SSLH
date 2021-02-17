
import json
import numpy as np
import os
import os.path as osp
import re

from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional


CROSS_VAL_COMMON_KEYS = [
	"run_name", "start_date", "tag", "dataset", "model", "seed", "supervised_ratio", "nb_epochs", "git_hash",
	"cross_val_tag",
]
DEFAULT_INCLUDE_LIST = [
	"^run_name$", "^start_date$", "^tag$", "^duration$",
]

FOLDS_VAL_KEY = "current_folds_val"
DURATION_KEY = "duration"


def create_args() -> Namespace:
	parser = ArgumentParser()

	parser.add_argument("--roots", "--tensorboard_roots", "-tr", type=str, nargs="*",
		default=[osp.join("..", "results", "tensorboard")],
		help="List of tensorboard roots directories paths. (default: ['../results/tensorboard/'])")

	parser.add_argument("--pattern", "-p", type=str, default=".*",
		help="Pattern for tensorboard paths. (default: '.*')")

	parser.add_argument("--exclude_list", "-ex", type=str, nargs="*", default=[],
		help="Exclude columns list. (default: [])")

	parser.add_argument("--include_list", "-in", type=str, nargs="*", default=[],
		help="Add the columns matching with at least one element of this list. (default: [])")

	parser.add_argument("--default_include_list", "-in_def", type=str, nargs="*", default=DEFAULT_INCLUDE_LIST,
		help="Another include list. Contains the default values printed without metrics. (default: {})".format(
			str(DEFAULT_INCLUDE_LIST)))

	parser.add_argument("--metrics", type=str, nargs="*", default=["val/acc", "eval/acc"],
		help="The name of the main metric to use for compare results. (default: ['val/acc', 'eval/acc'])")

	parser.add_argument("--criterions", type=str, nargs="*", default=["max"],
		help="The name of the criterion for the main metric. (default ['max'])")

	parser.add_argument("--cross_val", "-cv", action="store_true", default=False,
		help="Concatenate cross-validation results.")

	parser.add_argument("--result_index", "-ri", action="store_true", default=False,
		help="Print the index of the results at each line beginning.")

	parser.add_argument("--verbose", "-v", type=int, default=0,
		help="Verbose level of the program. (default: 0)")

	str_to_optional_str = lambda x: None if str(x).lower() == "none" else str(x)
	parser.add_argument("--float_format", "-ff", type=str_to_optional_str, default="{:.5f}",
		help="The format for floating types. (default: '{:.5f}')")

	args = parser.parse_args()

	if len(args.criterions) == 1:
		args.criterions = [args.criterions[0]] * len(args.metrics)

	if len(args.criterions) != len(args.metrics):
		raise RuntimeError("Criterions must have the same size than metrics or must contains 1 criterion that will be "
			"used for all metrics.")
	return args


def main():
	args = create_args()

	for root in args.roots:
		if not osp.isdir(root):
			raise RuntimeError(f"Unknown directory '{root}'.")

		results_paths = get_results_paths(root, args)
		results = []
		param_names = []

		for run_path in results_paths:
			objects = get_sub_paths(run_path)
			filepaths = [path for path in sorted(objects) if osp.isfile(path)]

			if args.verbose:
				print(f"Check results folder '{run_path}'...")

			run_content = {}
			for path in filepaths:
				if path.endswith(".json"):
					with open(path, "r") as file:
						content = json.load(file)
						run_content.update(content)

			run_results = {}
			for key in ("args", "info", "other"):
				if key in run_content.keys():
					intersection = list(set(run_results.keys()).intersection(run_content[key].keys()))
					if len(intersection) != 0:
						print(f"WARNING: Found common(s) key(s) between args, info and other json files. "
							f"('{intersection}' is not empty)")
					run_results.update(run_content[key])

			if "bests" in run_content.keys():
				if len(args.metrics) != len(args.criterions) and len(args.criterions) != 1:
					raise RuntimeError("Mismatch between metrics and criterions sizes.")

				for metric_name, criterion in zip(args.metrics, args.criterions):
					if metric_name in run_content["bests"]:
						metric_value = run_content["bests"][metric_name][criterion]
						run_results[metric_name] = metric_value

			for name in run_results.keys():
				if name not in param_names:
					param_names.append(name)

			# If cross-val mode is True, we only add runs with cross-validation activated.
			if not args.cross_val or ("cross_validation" in run_results.keys() and run_results["cross_validation"]):
				results.append(run_results)

		if not args.cross_val:
			param_names = [name for name in param_names if pass_filters(name, args)]
			matrix = build_matrix_values(results, param_names, args)
		else:
			matrix, param_names = build_matrix_cross_val(results, param_names, args)

		if len(matrix) == 0:
			print(f"No matching results found in root directory '{root}'.")
			continue

		# Format columns
		col_to_remove = set()
		for col_idx, name in enumerate(param_names):
			if not pass_filters(name, args):
				col_to_remove.add(col_idx)
				continue

			column = get_column(matrix, col_idx)
			content_max_len = max((len(value) for value in column))
			if content_max_len == 0:
				col_to_remove.add(col_idx)
				continue
			max_len = max(content_max_len, len(name))

			col_format = "{{:{:d}s}}".format(max_len)
			for row_idx in range(len(matrix)):
				matrix[row_idx][col_idx] = col_format.format(matrix[row_idx][col_idx])
			param_names[col_idx] = col_format.format(name)

		# Remove columns (reverse index sort because index will not be valid in deleted in another order.
		for col_idx in sorted(col_to_remove, reverse=True):
			param_names.pop(col_idx)
			for line in matrix:
				line.pop(col_idx)
				assert len(param_names) == len(line)

		print_matrix(param_names, matrix, args)


def get_sub_paths(root: str) -> List[str]:
	objects = os.listdir(root)
	objects = [osp.join(root, name) for name in objects]
	return objects


def get_results_paths(root: str, args: Namespace) -> List[str]:
	objects = get_sub_paths(root)
	objects = [path for path in objects if osp.isdir(path)]
	results_paths = [path for path in objects if re.search(args.pattern, path) is not None]
	return results_paths


def pre_process_value(value: Any, args: Namespace) -> str:
	if value is None:
		return ""
	elif isinstance(value, float) and args.float_format is not None:
		return args.float_format.format(value)
	else:
		return str(value)


def build_matrix_values(results: List[Dict[str, Any]], param_names: List[str], args: Namespace) -> List[List[str]]:
	return [
		[pre_process_value(run_results[name], args) if name in run_results else "" for name in param_names]
		for run_results in results
	]


def build_matrix_cross_val(
	results: List[Dict[str, Any]], param_names: List[str], args: Namespace
) -> (List[List[str]], List[str]):
	groups_run_indexes = []
	all_folds_val = []
	for run_index, run_results in enumerate(results):
		group_index = search_group(run_results, groups_run_indexes, results)
		if group_index is None:
			groups_run_indexes.append([run_index])
		else:
			groups_run_indexes[group_index].append(run_index)

		if FOLDS_VAL_KEY in run_results.keys() and run_results[FOLDS_VAL_KEY] not in all_folds_val:
			all_folds_val.append(run_results[FOLDS_VAL_KEY])

	all_folds_val.sort(key=lambda x: str(x))

	folds_metrics_names = []
	for name in args.metrics:
		for folds_val in all_folds_val:
			folds_metrics_names.append(f"{name}_F{str(folds_val)}")

	param_names_new = \
		CROSS_VAL_COMMON_KEYS + \
		[f"{DURATION_KEY}_sum"] + \
		[f"{name}_mean" for name in args.metrics] + \
		[f"{name}_std" for name in args.metrics] + \
		folds_metrics_names
	other_names = [name for name in param_names if name not in param_names_new]
	param_names_new = param_names_new + other_names

	matrix = []
	for group_run_indexes in groups_run_indexes:
		if len(group_run_indexes) == 0:
			raise RuntimeError("Found an empty group of cross-validation runs.")

		first_run_results = results[group_run_indexes[0]]
		common_values = [first_run_results[key] if key in first_run_results else "" for key in CROSS_VAL_COMMON_KEYS]
		durations = [results[idx][DURATION_KEY] if DURATION_KEY in results[idx] else "" for idx in group_run_indexes]
		durations_sum = [sum(durations)] if "" not in durations else [""]

		metrics_values = {
			metric: [
				results[idx][metric] if metric in results[idx] else None
				for idx in group_run_indexes
			]
			for metric, criterion in zip(args.metrics, args.criterions)
		}
		metrics_means = [
			np.mean(values) if all((val is not None) for val in values) else "" for values in metrics_values.values()]
		metrics_stds = [
			np.std(values) if all((val is not None) for val in values) else "" for values in metrics_values.values()]

		folds_values = []
		for name, values in metrics_values.items():
			metric_folds_values = [""] * len(all_folds_val)
			for i, idx in enumerate(group_run_indexes):
				metric_value = values[i]
				if FOLDS_VAL_KEY not in results[idx]:
					continue

				run_folds_val = results[idx][FOLDS_VAL_KEY]
				if run_folds_val not in all_folds_val:
					raise RuntimeError(f"Unknown folds_val list '{str(run_folds_val)}' in group folds val '{str(all_folds_val)}'.")

				index_folds = all_folds_val.index(run_folds_val)
				if metric_folds_values[index_folds] != "":
					raise RuntimeError("Found the same fold twice in a group.")

				metric_folds_values[index_folds] = metric_value

			folds_values.extend(metric_folds_values)

		line_values = common_values + durations_sum + metrics_means + metrics_stds + folds_values

		for name in other_names:
			values = [results[idx][name] if name in results[idx] else "" for idx in group_run_indexes]
			all_equals = values.count(values[0]) == len(values)
			if all_equals:
				line_values.append(values[0])
			else:
				line_values.append("")

		line_values = [pre_process_value(value, args) for value in line_values]
		matrix.append(line_values)

	assert len(matrix) == 0 or len(matrix[0]) == len(param_names_new)
	return matrix, param_names_new


def search_group(
	run_results: Dict[str, Any], groups_run_indexes: List[List[int]], results: List[Dict[str, Any]]
) -> Optional[int]:
	for group_index, group_run_indexes in enumerate(groups_run_indexes):
		first_results_group = results[group_run_indexes[0]]
		if are_in_same_group(run_results, first_results_group):
			return group_index
	return None


def are_in_same_group(run_results_1: Dict[str, Any], run_results_2: Dict[str, Any]) -> bool:
	return all((
		key in run_results_1.keys() and key in run_results_2.keys() and run_results_1[key] == run_results_2[key]
		for key in CROSS_VAL_COMMON_KEYS
	))


def get_column(matrix: List[List[str]], col_idx: int) -> List[str]:
	return [row[col_idx] for row in matrix]


def pass_filters(name: str, args: Namespace) -> bool:
	include_list = set(args.include_list).union(args.metrics).union(args.default_include_list)
	included = len(include_list) == 0 or any([re.search(include, name) is not None for include in include_list])

	exclude_list = set(args.exclude_list)
	excluded = len(exclude_list) > 0 and any([re.search(exclude, name) is not None for exclude in exclude_list])
	return included and not excluded


def print_matrix(param_names: List[str], matrix: List[List[str]], args: Namespace):
	if args.result_index:
		num_len = len(str(len(matrix)))
		print("| {} | {} |".format(" " * num_len, " | ".join(param_names)))
		for num, line in enumerate(matrix):
			num_format = f"{{:{num_len}d}}"
			print("| {} | {} |".format(num_format.format(num), " | ".join(line)))
	else:
		print("| {} |".format(" | ".join(param_names)))
		for num, line in enumerate(matrix):
			print("| {} |".format(" | ".join(line)))


if __name__ == "__main__":
	main()
