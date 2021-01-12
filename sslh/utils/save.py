
import csv
import json
import os.path as osp

from argparse import Namespace
from sslh.utils.recorder.base import RecorderABC
from sslh.utils.misc import to_dict_rec
from time import time
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Callable, Dict, List, Union


def save_results(
	results_dirpath: str,
	args: Namespace, recorder: RecorderABC,
	augments_dict: Dict[str, Callable],
	main_metric_name: str,
	start_date: str,
	folds_val: List[int],
	start_time: float,
):
	"""
		Save results in different files in tensorboard logdir.
	"""
	if args.write_results:
		print("Results will be saved in tensorboard writer \"{:s}\".".format(results_dirpath))
		duration = time() - start_time

		writer = recorder.get_writer()
		if writer is not None:
			save_writer(args, recorder, augments_dict, writer, duration)

		filepath_args = osp.join(results_dirpath, "args.json")
		save_args(filepath_args, args)

		filepath_augms = osp.join(results_dirpath, "callables.json")
		save_augms(filepath_augms, augments_dict)

		filepath_best_epoch = osp.join(results_dirpath, "best_epoch.json")
		save_best_values(filepath_best_epoch, recorder)

		filepath_static = osp.join(results_dirpath, "static.json")
		save_static(filepath_static, args, folds_val, recorder, duration)


def save_best_values(filepath: str, recorder: RecorderABC):
	content = {}
	names = recorder.get_all_names()
	for name in names:
		idx_min, min_, idx_max, max_ = recorder.get_min_max(name)
		content[name] = {"min": min_, "idx_min": idx_min, "max": max_, "idx_max": idx_max}
	with open(filepath, "w") as file:
		json.dump(content, file, indent="\t")


def save_writer(
	args: Namespace,
	recorder: RecorderABC,
	augments_dict: Dict[str, Callable],
	writer: SummaryWriter,
	duration: float,
):
	hparam_dict = dict(args.__dict__)
	metric_dict = {}
	for name in recorder.get_all_names():
		idx_min, min_, idx_max, max_ = recorder.get_min_max(name)
		section, sub_name = name.split("/") if "/" in name else ("", name)

		for suffix, value in zip(
			("idx_min", "min", "idx_max", "max"),
			(idx_min, min_, idx_max, max_)
		):
			new_name = f"{section}_best/{sub_name}_{suffix}"
			metric_dict[new_name] = value

	def _convert_value(v: Any) -> Union[int, float, str, bool, Tensor]:
		if not(isinstance(v, (int, float, str, bool, Tensor))):
			return str(v)
		else:
			return v

	hparam_dict = {str(k): _convert_value(v) for k, v in hparam_dict.items()}
	metric_dict = {str(k): _convert_value(v) for k, v in metric_dict.items()}

	writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
	writer.add_text("args", json.dumps(args.__dict__, indent="\t"))
	writer.add_text("duration", str(duration))
	try:
		transforms_dump = json.dumps(to_dict_rec(augments_dict), indent="\t")
		writer.add_text("transforms", transforms_dump)
	except TypeError:
		print("Warning: Cannot save callables in JSON file.")

	writer.flush()
	writer.close()


def save_args(filepath: str, args: Namespace):
	"""
		Save arguments in JSON file.

		:param filepath: The filepath where to save the arguments.
		:param args: argparse arguments.
	"""
	content = {"args": args.__dict__}
	with open(filepath, "w") as file:
		json.dump(content, file, indent="\t")


def save_augms(filepath: str, augms: Any):
	"""
		Save callables to JSON file.

		:param filepath: The path to JSON file.
		:param augms: A dictionary or list of callables used.
	"""
	content = {"callables": to_dict_rec(augms, "__class__")}
	with open(filepath, "w") as file:
		json.dump(content, file, indent="\t")


def save_static(filepath: str, args: Namespace, folds_val: List[int], recorder: RecorderABC, duration: float):
	content = {
		"dataset": args.dataset_name,
		"git_hash": args.git_hash,
		"start_date": args.start_date,
		"run_name": args.run_name,
		"folds_val": folds_val,
		"duration": duration,
	}
	with open(filepath, "w") as file:
		json.dump(content, file, indent="\t")
