
import json
import os.path as osp
import random

from argparse import ArgumentParser, Namespace
from sslh.utils.types import (
	str_to_bool, str_to_optional_str, str_to_optional_float, str_to_optional_int, float_in_range, str_to_optional_bool
)


def add_common_args(parser: ArgumentParser) -> ArgumentParser:
	"""
		Add common arguments in main scripts to an ArgumentParser.
	"""
	# ------------------
	# Main args
	# ------------------
	group_main = parser.add_argument_group("Main args")

	group_main.add_argument("--dataset_name", "--dataset", type=str, default="ESC10",
		choices=["CIFAR10", "UBS8K", "ESC10", "ESC50", "GSC", "GSC12", "AUDIOSET"],
		help="Dataset name to use. (default: 'ESC10')")

	group_main.add_argument("--dataset_path", "--dataset_root", type=str, default=osp.join("..", "data"),
		help="Dataset root dir where the data is stored. (default: '../data')")

	group_main.add_argument("--seed", type=int, default=1234,
		help="Seed for random generators. (default: 1234)")

	group_main.add_argument("--debug", "--debug_mode", type=str_to_bool, default=False,
		help="Debugging mode for detect errors. Can be slower than normal mode. (default: False)")

	group_main.add_argument("--tag", "--suffix", type=str, default="",
		help="Tag used in training name and as suffix to tensorboard log dir. (default: '')")

	group_main.add_argument("--logdir", "--tensorboard_path", "--tensorboard_root", type=str,
		default=osp.join("..", "results", "tensorboard"),
		help="Tensorboard parent directory (logdir). (default: '../results/tensorboard')")

	group_main.add_argument("--nb_epochs", "--epochs", type=int, default=300,
		help="Number of epoch to run. (default: 300)")

	group_main.add_argument("--write_results", "--save_results", type=str_to_bool, default=True,
		help="Write results in a tensorboard SummaryWriter. (default: True)")

	group_main.add_argument("--args_filepaths", type=str, nargs="+", default=None,
		help="List of filepaths to arguments file. Values in this JSON will overwrite other options in terminal. "
		"(default: None)")

	group_main.add_argument("--cross_val_tag", type=str, default=hex(random.randint(0, 2**32)),
		help="A tag for mark runs that are part of the same cross-validation process.")

	group_main.add_argument("--checkpoint_path", type=str, default=osp.join("..", "results", "models"),
		help="Directory path where checkpoint models will be saved. (default: '../results/models')")

	group_main.add_argument("--supervised_ratio", "--su_ratio", type=float, default=0.1,
		help="Supervised ratio used for split dataset. (default: 0.1)")

	group_main.add_argument("--cross_validation", "--cross_val", "--cv", type=str_to_bool, default=False,
		help="Use cross validation for UBS8K dataset. (default: False)")

	group_main.add_argument("--folds_train", type=str_to_optional_int, nargs="+", default=None,
		help="Fold used for training in ESC10 or UBS8K dataset. "
		"This parameter is unused if cross validation_old is True or on other datasets. (default: None)")

	group_main.add_argument("--folds_val", type=str_to_optional_int, nargs="+", default=None,
		help="Fold used for validation in ESC10 or UBS8K dataset. "
		"This parameter is unused if cross validation_old is True or on other datasets. (default: None)")

	group_main.add_argument("--ra_nb_choices", type=int, default=1,
		help="Nb augmentations composed for RandAugment. (default: 1)")

	group_main.add_argument("--ra_pool", type=str_to_optional_str, default="pool1",
		help="RandAugment augment pool name. (default: 'pool1')")

	group_main.add_argument("--label_smoothing_value", "--smooth", type=float, default=0.0,
		help="Label smoothing value for supervised trainings. Use 0.0 for deactivate label smoothing. "
		"If value is 1.0, all labels will be a uniform distribution. (default: 0.0)")

	group_main.add_argument("--batch_size_s", "--bsize_s", "--bsize", "--bs", type=int, default=30,
		help="Batch size used for supervised loader. (default: 30)")

	group_main.add_argument("--zip_cycle_policy", "--zip_policy", type=str, default="max",
		choices=["min", "max"],
		help="Criterion of ZipCycle end. If Max, stop when the loader with the higher length is finished, otherwise "
		"stop when the loader with the lower length is finished. (default 'max')")

	group_main.add_argument("--train_version", type=str, default="unbalanced",
		choices=["balanced", "unbalanced"],
		help="The AudioSet train subset used. (default: 'unbalanced')")

	group_main.add_argument("--nb_gpu", type=int, default=1,
		help="Number of GPU to use. (default: 1)")

	# ------------------
	# Model args
	# ------------------
	group_model = parser.add_argument_group("Model args")
	group_model.add_argument("--model", type=str_to_optional_str, default="WideResNet28",
		help="Model to run. If None, select the default model for the dataset chosen. (default: 'WideResNet28')")

	group_model.add_argument("--dropout", type=float_in_range(0.0, 1.0), default=0.0,
		help="Dropout used in model. WARNING: All models does not use this dropout argument. (default: 0.0)")

	group_model.add_argument("--wrn_width", type=int, default=2,
		help="Width for WideResNet model. (default: 2)")

	group_model.add_argument("--wrn_num_input_channels", type=int, default=1,
		help="Number of input channels for WideResNet. (default: 1)")

	group_model.add_argument("--device", "--device_name", type=str, default="cuda",
		help="Device name used for training. (default: 'cuda')")

	group_model.add_argument("--activation", "--acti", type=str, default="softmax",
		help="The activation function to use for the model. (default: 'softmax')")

	# ------------------
	# Optimizer args
	# ------------------
	group_optim = parser.add_argument_group("Optimizer args")
	group_optim.add_argument("--optimizer", "--optim", type=str, default="Adam",
		choices=["Adam", "SGD", "RAdam", "PlainRAdam", "AdamW"],
		help="Optimizer used. (default: 'Adam')")

	group_optim.add_argument("--learning_rate", "--lr", type=float, default=1e-3,
		help="Learning rate to use. (default: 1e-3)")

	group_optim.add_argument("--weight_decay", "--wd", type=str_to_optional_float, default=None,
		help="Weight decay used. Use None for use the default weight decay of the optimizer. (default: None)")

	group_optim.add_argument("--momentum", type=str_to_optional_float, default=None,
		help="Momentum used in SGD optimizer. Use None for use the default momentum of the optimizer. (default: None)")

	group_optim.add_argument("--use_nesterov", "--nesterov", type=str_to_optional_bool, default=False,
		help="Activate Nesterov momentum for SGD optimizer. Use None for use the default momentum of the optimizer. "
		"(default: False)")

	# ------------------
	# Scheduler args
	# ------------------
	group_sched = parser.add_argument_group("Scheduler args")
	group_sched.add_argument("--scheduler", "--sched", type=str_to_optional_str, default=None,
		choices=[
			None,
			"CosineLRScheduler", "cos",
			"MultiStepLR", "step",
			"SoftCosineLRScheduler", "softcos",
		],
		help="FixMatch scheduler used. Use None for not using any scheduler. (default: None)")

	group_sched.add_argument("--lr_decay_ratio", type=float, default=0.2,
		help="Learning rate decay ratio used in MultiStepLR scheduler. (default: 0.2)")

	group_sched.add_argument("--epoch_steps", type=int, nargs="+", default=[60, 120, 160],
		help="Epochs where we decrease the learning rate. Used in MultiStepLR scheduler. (default: [60, 120, 160])")

	group_sched.add_argument("--sched_coef", type=float, default=7.0 / 16.0,
		help="Coefficient used in CosineLRScheduler. (default: 7/16)")

	group_sched.add_argument("--sched_nb_steps", type=str_to_optional_float, default=None,
		help="The number of steps in CosineLRScheduler. If None, use the number of epochs. (default: None)")

	return parser


def check_args(args: Namespace):
	"""
		Check arguments (mainly directories and files)

		:param args: argparse arguments.
	"""
	if args.write_results:
		if not osp.isdir(args.logdir):
			raise RuntimeError(f"Invalid dirpath '{args.logdir}'.")
		if not osp.isdir(args.checkpoint_path):
			raise RuntimeError(f"Invalid dirpath '{args.checkpoint_path}'.")

	if args.args_filepaths is not None:
		for filepath in args.args_filepaths:
			if not osp.isfile(filepath):
				raise RuntimeError(f"Invalid filepath '{filepath}'.")

	if not osp.isdir(args.dataset_path):
		raise RuntimeError(f"Invalid dirpath '{args.dataset_path}'.")


def post_process_args(args: Namespace) -> Namespace:
	"""
		Update arguments by adding some parameters inside.

		:param args: argparse arguments.
		:returns: The updated argparse arguments.
	"""
	if hasattr(args, "args_filepaths"):
		args_filepaths = args.args_filepaths

		if args.args_filepaths is not None:
			for filepath in args.args_filepaths:
				args = load_args(filepath, args)

		args.args_filepaths = args_filepaths

	return args


def load_args(filepath: str, args: Namespace, check_keys: bool = True) -> Namespace:
	"""
		Load arguments from a JSON file.

		:param filepath: The path to JSON file.
		:param args: argparse arguments to update.
		:param check_keys: If True, check if keys of JSON file are inside args keys.
		:returns: The argparse arguments updated.
	"""
	with open(filepath, "r") as file:
		file_dict = json.load(file)
		if "args" not in file_dict.keys():
			raise RuntimeError("Invalid args file or too old args file version.")

		args_file_dict = file_dict["args"]

		if check_keys:
			differences = set(args_file_dict.keys()).difference(args.__dict__.keys())
			if len(differences) > 0:
				raise RuntimeError("Found unknown(s) key(s) in JSON file : '{:s}'.".format(", ".join(differences)))

		args.__dict__.update(args_file_dict)

		# Post process : convert "none" strings to None value
		for name, value in args.__dict__.items():
			if isinstance(value, str) and value.lower() == "none":
				args.__dict__[name] = None

	return args
