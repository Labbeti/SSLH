
import csv
import json
import os.path as osp

from argparse import Namespace
from sslh.utils.recorder.recorder_abc import RecorderABC
from sslh.utils.misc import to_dict_rec
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
):
	"""
		Save results in different files in tensorboard logdir.
	"""
	if args.write_results:
		print("Results will be saved in tensorboard writer \"{:s}\".".format(results_dirpath))

		writer = recorder.get_writer()
		if writer is not None:
			save_writer(args, recorder, augments_dict, writer)

		filepath_args = osp.join(results_dirpath, "args.json")
		save_args(filepath_args, args)

		filepath_augms = osp.join(results_dirpath, "callables.json")
		save_augms(filepath_augms, augments_dict)

		filepath_best_epoch = osp.join(results_dirpath, "best_epoch.json")
		recorder.save_in_file(filepath_best_epoch)

		# TODO : rem this save ?
		filepath_results_csv = osp.join(results_dirpath, "results.csv")
		save_csv(filepath_results_csv, args, recorder, main_metric_name, start_date)

		filepath_static = osp.join(results_dirpath, "static.json")
		save_static(filepath_static, args, folds_val, recorder)


def save_writer(
	args: Namespace,
	recorder: RecorderABC,
	augments_dict: Dict[str, Callable],
	writer: SummaryWriter,
):
	hparam_dict = dict(args.__dict__)
	metric_dict = {}
	for key, best_values in recorder.get_all_best_epoch().items():
		section, metric_name = key.split("/")
		# section = train, val or eval
		key = "{:s}_best/{:s}".format(section, metric_name)
		for best_type, value in best_values.items():
			# best_type = best_epoch, best_mean or best_std
			new_key = "{:s}_{:s}".format(key, best_type)
			metric_dict[new_key] = value

	def _convert_value(v: Any) -> Union[int, float, str, bool, Tensor]:
		if not(isinstance(v, (int, float, str, bool, Tensor))):
			return str(v)
		else:
			return v

	hparam_dict = {str(k): _convert_value(v) for k, v in hparam_dict.items()}
	metric_dict = {str(k): _convert_value(v) for k, v in metric_dict.items()}

	writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
	writer.add_text("args", json.dumps(args.__dict__, indent="\t"))
	writer.add_text("duration", str(recorder.get_elapsed_time()))
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


def save_csv(filepath: str, args: Namespace, recorder: RecorderABC, main_metric_name: str, start_date: str):
	best = recorder.get_best_epoch(main_metric_name)
	with open(filepath, "w") as file:
		writer = csv.writer(file)
		columns_names = [
			main_metric_name,
			main_metric_name + "_std",
			"",
			"su_ratio",
			"nb_epochs",
			"optim",
			"sched",
			"nb_augms",
			"threshold",
			"bsize_s",
			"bsize_u",
			"temperature",
			"seed",
			"lr",
			"start_date",
		]
		values = [
			best["best_mean"],
			best["best_std"],
			"",
			args.supervised_ratio,
			args.nb_epochs,
			args.optimizer,
			args.scheduler if args.scheduler is not None else "",
			args.nb_augms if hasattr(args, "nb_augms") else "",
			args.threshold if hasattr(args, "threshold") else "",
			args.batch_size_s,
			args.batch_size_u if hasattr(args, "batch_size_u") else "",
			args.temperature if hasattr(args, "temperature") else "",
			args.seed,
			args.learning_rate,
			start_date,
		]
		writer.writerows([columns_names, values])


def save_static(filepath: str, args: Namespace, folds_val: List[int], recorder: RecorderABC):
	content = {
		"dataset": args.dataset_name,
		"git_hash": args.git_hash,
		"start_date": args.start_date,
		"train_name": args.train_name,
		"folds_val": folds_val,
		"duration": recorder.get_elapsed_time(),
	}
	with open(filepath, "w") as file:
		json.dump(content, file, indent="\t")
