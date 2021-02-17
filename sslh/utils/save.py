
import json
import os.path as osp
import re
import torch

from argparse import Namespace

from sslh.datasets.get_builder import DatasetBuilder
from sslh.utils.recorder.base import RecorderABC

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional, Union


def save_results_files(
	results_dirpath: str,
	run_name: str,
	duration: float,
	start_date: str,
	git_hash: str,
	folds_train: Optional[List[int]],
	folds_val: Optional[List[int]],
	builder: Optional[DatasetBuilder],
	args: Optional[Namespace],
	recorder: Optional[RecorderABC],
	other: Optional[Dict[str, Any]] = None,
):
	if not args.write_results:
		return

	if not osp.isdir(results_dirpath):
		raise RuntimeError(f"Cannot save results because '{results_dirpath}' is not a directory.")

	duration = int(duration)
	dataset = builder.get_dataset_name() if builder is not None else "UNKNOWN"

	info = {
		"run_name": run_name,
		"start_date": start_date,
		"duration": duration,
		"dataset": dataset,
		"git_hash": git_hash,
		# Note : cannot call the folds "folds_train" and "folds_val" because it is already a argument name.
		"current_folds_train": folds_train,
		"current_folds_val": folds_val,
	}
	content = {"info": info}
	save_in_json(content, "info.json", results_dirpath)

	writer = recorder.get_writer()
	if writer is not None:
		save_data_in_writer(writer, args, recorder, info)

	if args is not None:
		content = {"args": args.__dict__}
		save_in_json(content, "args.json", results_dirpath)

	if recorder is not None:
		min_max = {}
		names = recorder.get_all_names()
		for name in names:
			idx_min, min_, idx_max, max_ = recorder.get_min_max(name)
			min_max[name] = {"idx_min": idx_min, "min": min_, "idx_max": idx_max, "max": max_}
		content = {"bests": min_max}
		save_in_json(content, "bests.json", results_dirpath)

	if other is not None:
		content = {"other": other}
		save_in_json(content, "other.json", results_dirpath)


def save_data_in_writer(
	writer: SummaryWriter,
	args: Optional[Namespace],
	recorder: Optional[RecorderABC],
	info: Dict[str, Any],
):
	hparam_dict = args.__dict__ if args is not None else {}
	metric_dict = {}
	all_min_max = recorder.get_all_min_max()
	for name, min_max in all_min_max.items():
		section, sub_name = name.split("/") if "/" in name else ("", name)
		for suffix, value in min_max.items():
			new_name = f"{section}_best/{sub_name}_{suffix}"
			metric_dict[new_name] = value

	def convert_value(v: Any) -> Union[int, float, str]:
		if isinstance(v, (int, float, str)):
			return v
		elif isinstance(v, Tensor) and torch.prod(torch.as_tensor(v.shape, dtype=torch.int)) == 1:
			return v.item()
		else:
			return str(v)

	hparam_dict = {str(k): convert_value(v) for k, v in hparam_dict.items()}
	metric_dict = {str(k): convert_value(v) for k, v in metric_dict.items()}

	writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

	args_dumps = json.dumps(args.__dict__, indent="\t") if args is not None else ""
	writer.add_text("args", args_dumps)

	for name, value in info.items():
		writer.add_text(name, str(value))

	writer.flush()
	writer.close()


def save_in_json(content: Dict[str, Any], filepath: str, dirpath: Optional[str]):
	if dirpath is not None:
		filepath = osp.join(dirpath, filepath)
	if osp.exists(filepath):
		raise RuntimeError(f"Cannot save results, object '{filepath} already exists.")
	with open(filepath, "w") as file:
		json.dump(content, file, indent="\t")


def duration_formatter(seconds: float, format_: str = "%jd:%Hh:%Mm:%Ss") -> str:
	rest = int(seconds)

	rest, seconds = divmod(rest, 60)
	rest, minutes = divmod(rest, 60)
	rest, hours = divmod(rest, 24)
	days = rest

	replaces = {
		"%S": seconds,
		"%M": minutes,
		"%H": hours,
		"%j": days,
	}
	result = format_
	for directive, value in replaces.items():
		result = result.replace(directive, str(value))
	return result


def duration_unformatter(string: str, format_: str = "%jd:%Hh:%Mm:%Ss") -> float:
	replaces = {
		"%S": "(?P<S>[0-9]+)",
		"%M": "(?P<M>[0-9]+)",
		"%H": "(?P<H>[0-9]+)",
		"%j": "(?P<j>[0-9]+)",
	}
	format_re = format_
	for directive, value in replaces.items():
		format_re = format_re.replace(directive, str(value))

	match = re.search(format_re, string)
	if match is None:
		raise RuntimeError(f"Invalid string '{string}' with format '{format_}'.")

	seconds = int(match["S"])
	minutes = int(match["M"])
	hours = int(match["H"])
	days = int(match["j"])
	total_seconds = seconds + minutes * 60 + hours * 3600 + days * 3600 * 24
	return total_seconds
