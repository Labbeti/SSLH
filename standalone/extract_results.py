
import json
import numpy as np
import os
import os.path as osp
import re

from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional


RUN_NAME = "Extract"
NAME_KEY = "Nom"


def create_args() -> Namespace:
	parser = ArgumentParser()

	parser.add_argument("--tensorboard_roots", "-tr", type=str, nargs="+", default=[osp.join("..", "results", "tensorboard")],
		help="List of tensorboard roots directories paths. (default: [\"../results/tensorboard/\"])")

	parser.add_argument("--pattern", "-p", type=str, default=".*",
		help="Pattern for tensorboard paths. (default: \".*\")")

	parser.add_argument("--verbose", "-v", type=int, default=0,
		help="Verbose level of the program. (default: 0)")

	parser.add_argument("--no_empty_column", "-ne", type=lambda x: x.lower() in ["1", "y", "yes", "true"], default=True,
		help="Remove columns where all values are empty strings. (default: True)")

	parser.add_argument("--search_cross_val", "-cv", action="store_true", default=False,
		help="Concatenate results of cross-validations. (default: False)")

	parser.add_argument("--sort_keys", "-s", action="store_true", default=False,
		help="Sort columns by name. (default: False)")
	
	parser.add_argument("--exclude_list", "-ex", type=str, nargs="+", default=[],
		help="Exclude columns list. (default: [])")
	parser.add_argument("--include_list", "-in", type=str, nargs="+", default=[],
		help="If contains at least 1 element, will only print the Name and the elements of this list. (default: [])")

	args = parser.parse_args()

	return args


def get_train_name_from_dirpath(dirpath: str) -> str:
	dirname = osp.basename(dirpath)
	train_names = [
		"FixMatch", "FixMatchExp", "MixMatch", "MixMatchExp", "ReMixMatch", "SupervisedAugment", "Supervised", "SupervisedExp", "UDA", "UDAExp"
	]
	for name in train_names:
		name_with_underscores = f"_{name}_"
		if name_with_underscores in dirname:
			return name

	raise RuntimeError(f"Cannot find train name in dirname {dirname}.")


def get_group(results: dict, args_dict: dict, groups: List[list]) -> Optional[int]:
	for group_idx, group in enumerate(groups):
		_idx, group_res, group_args, _best_dict = group[0]
		if args_dict == group_args:
			return group_idx
	return None


def get_folds_from_dirpath(dirpath: str) -> Optional[List[int]]:
	dirname = osp.basename(dirpath)
	pattern_fold = r"\[(?P<folds>[\d|,| ]*)\]"
	match = re.search(pattern_fold, dirname)

	if match is None:
		return None
	else:
		folds = match["folds"]
		folds = folds.replace(" ", "")
		folds = folds.split(",")
		folds = [int(fold) for fold in folds]
		return folds


def get_keys(global_results: list) -> List[str]:
	keys = []
	for results in global_results:
		for key in results.keys():
			if key not in keys:
				keys.append(key)
	return keys


def get_key_formats(keys: List[str], global_results: list, args: Namespace) -> Dict[str, str]:
	max_lengths = {
		key: max([len(str(results[key])) if key in results.keys() else 0 for results in global_results])
		for key in keys
	}

	to_remove = []
	for key, max_len in max_lengths.items():
		if key != NAME_KEY and (
			(args.no_empty_column and max_len == 0) or
			(len(args.exclude_list) > 0 and any([re.search(exclude, key) for exclude in args.exclude_list])) or
			(len(args.include_list) > 0 and all([not re.search(include, key) for include in args.include_list]))
		):
			to_remove.append(key)

	for key in to_remove:
		max_lengths.pop(key)
		keys.remove(key)
		for results in global_results:
			if key in results.keys():
				results.pop(key)

	max_lengths = {key: max(max_len, len(key)) for key, max_len in max_lengths.items()}
	key_formats = {key: "{{:{:d}s}}".format(max_len) for key, max_len in max_lengths.items()}
	return key_formats


def print_global_results(global_results: list, args: Namespace):
	if len(global_results) == 0:
		print("No results found.")
		return

	lambda_sort = \
		lambda results_: "{:s}_{:s}".format(results_[NAME_KEY], results_["tag"]) if "tag" in results_.keys() else results_[NAME_KEY]

	keys = get_keys(global_results)
	key_formats = get_key_formats(keys, global_results, args)

	if args.sort_keys:
		keys.sort()

	# Print columns names
	end = " | "
	for key in keys:
		print(key_formats[key].format(str(key)), end=end)
	print()

	# Print values
	for results in sorted(global_results, key=lambda_sort):
		for key in keys:
			val = str(results[key]) if key in results.keys() else ""
			print(key_formats[key].format(val), end=end)
		print()


def main():
	args = create_args()

	for tensorboard_root in args.tensorboard_roots:
		objects = os.listdir(tensorboard_root)
		objects = [osp.join(tensorboard_root, name) for name in objects]
		objects = [path for path in objects if osp.isdir(path)]

		matches = [path for path in objects if re.search(args.pattern, path) is not None]
		# date_pattern = r"(?P<date>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})"

		# Read info from args and best epoch files
		global_results = []
		global_args_dict = []
		global_best_dict = []
		global_dirpaths = []

		for match in matches:
			if args.verbose >= 1:
				print(f"Match: \"{match}\" : ")
			args_filepath = osp.join(match, "args.json")
			best_epoch_filepath = osp.join(match, "best_epoch.json")
			static_filepath = osp.join(match, "static.json")

			if not osp.isfile(args_filepath) or not osp.isfile(best_epoch_filepath):
				if args.verbose >= 1:
					print(f"Warn: Ignore match \"{match}\" because cannot find args or best epoch JSON file(s).")
				continue

			with open(args_filepath, "r") as file:
				args_info = json.load(file)
			with open(best_epoch_filepath, "r") as file:
				best_epoch_info = json.load(file)

			if osp.isfile(static_filepath):
				with open(static_filepath, "r") as file:
					static_info = json.load(file)
			else:
				static_info = {}

			args_dict = args_info["args"]
			best_dict = best_epoch_info

			get_if_has = lambda dic, key: dic[key] if key in dic.keys() else ""

			folds_val = get_folds_from_dirpath(match)
			results = {
				"Nom": get_train_name_from_dirpath(match),
				# "Version": "",
				# "Gain": "",
				"Best val/acc": "{:.2f}%".format(best_dict["val/acc"]["max"] * 100.0) if "val/acc" in best_dict.keys() else "",
				"Best eval/acc": "{:.2f}%".format(best_dict["eval/acc"]["max"] * 100.0) if "eval/acc" in best_dict.keys() else "",
				"alpha": args_dict["alpha"] if "alpha" in args_dict.keys() else "",
				"su_ratio": "{:.2f}".format(args_dict["supervised_ratio"] * 100.0),
				"nb_epochs": args_dict["nb_epochs"], "optim": args_dict["optimizer"],
				"sched": args_dict["scheduler"] if args_dict["scheduler"] is not None else "",
				"nb_augms": args_dict["nb_augms"] if "nb_augms" in args_dict.keys() else "",
				"threshold": args_dict["threshold"] if "threshold" in args_dict.keys() else "",
				"bsize_s": args_dict["batch_size_s"],
				"bsize_u": args_dict["batch_size_u"] if "batch_size_u" in args_dict.keys() else "",
				"temperature": args_dict["temperature"] if "temperature" in args_dict.keys() else "",
				"seed": args_dict["seed"],
				"lr": args_dict["learning_rate"] if "learning_rate" in args_dict.keys() else "",
				"start_date": args_dict["start_date"] if "start_date" in args_dict.keys() else "",
				"tag": get_if_has(args_dict, "tag"),
				"folds_val": str(folds_val) if folds_val is not None else "",
				"duration": get_if_has(static_info, "duration"),
			}
			params = ["augm_none", "augm_weak", "augm_strong", "ema_decay", "weight_decay", "momentum", "use_nesterov"]
			for key in params:
				results[key] = get_if_has(args_dict, key)

			global_results.append(results)
			global_args_dict.append(args_dict)
			global_best_dict.append(best_dict)
			global_dirpaths.append(match)

		if not args.search_cross_val:
			print_global_results(global_results, args)
		else:
			groups = []
			for i, (results, args_dict, best_dict) in enumerate(zip(global_results, global_args_dict, global_best_dict)):
				group_idx = get_group(results, args_dict, groups)
				info = (i, results, args_dict, best_dict)
				if group_idx is not None:
					groups[group_idx].append(info)
				else:
					groups.append([info])

			groups = [group for group in groups if len(group) > 1]

			global_results = []
			for group in groups:
				_, first_results, _, _ = group[0]
				group_results = dict(first_results)

				val_acc = [best_dict["val/acc"]["best_mean"] for _, _, _, best_dict in group]
				durations = [results["duration"] for _, results, _, _ in group if results["duration"] != ""]

				group_results["Mean acc"] = "{:.2f}%".format(np.mean(val_acc) * 100.0)
				group_results["Std acc"] = "{:.2f}%".format(np.std(val_acc) * 100.0)
				group_results["Sum duration"] = np.sum(durations) if len(durations) == len(group) else ""

				group_results.pop("Best val/acc")
				group_results.pop("Best eval/acc")
				group_results.pop("folds_val")
				group_results.pop("duration")

				for idx, _, _, best_dict in group:
					dirpath = global_dirpaths[idx]
					folds_val = get_folds_from_dirpath(dirpath)
					group_results[f"Acc fold {str(folds_val)}"] = "{:.2f}%".format(best_dict["val/acc"]["best_mean"] * 100.0)

				global_results.append(group_results)

			print_global_results(global_results, args)


if __name__ == "__main__":
	main()
