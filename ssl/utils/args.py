import json
import os.path as osp
import subprocess

from argparse import ArgumentParser, Namespace
from ssl.utils.types import (
	str_to_bool, str_to_optional_str, str_to_optional_float, str_to_optional_int, float_in_range, str_to_optional_bool
)


def add_common_args(parser: ArgumentParser) -> ArgumentParser:
	"""
		Add common arguments in main scripts to an ArgumentParser.
	"""
	group = parser.add_argument_group("Main options")

	group.add_argument("--dataset_name", "--dataset", type=str, default="CIFAR10",
		choices=["CIFAR10", "UBS8K", "ESC10", "ESC50", "GSC", "GSC12", "AUDIOSET"],
		help="Dataset name to use. (default: CIFAR10)")

	group.add_argument("--dataset_path", "--dataset_root", type=str, default=osp.join("..", "datasets"),
		help="Dataset root dir where the data is stored. (default: \"../dataset\")")

	group.add_argument("--seed", type=int, default=1234,
		help="Seed for random generators. (default: 1234)")

	group.add_argument("--debug_mode", type=str_to_bool, default=False,
		help="Debugging mode for detect errors. Can be slower than normal mode. (default: False)")

	group.add_argument("--tag", "--suffix", type=str, default="",
		help="Tag used in training name and as suffix to tensorboard log dir. (default: \"\")")

	group.add_argument("--logdir", "--tensorboard_path", type=str, default=osp.join("..", "results", "tensorboard"),
		help="Tensorboard parent directory (logdir). (default: \"../results/tensorboard\")")

	group.add_argument("--nb_epochs", "--epochs", type=int, default=200,
		help="Number of epoch to run. (default: 200)")

	group.add_argument("--write_results", "--save_results", type=str_to_bool, default=True,
		help="Write results in a tensorboard SummaryWriter. (default: True)")

	group.add_argument("--args_filepaths", type=str, nargs="+", default=None,
		help="List of filepaths to arguments file. Values in this JSON will overwrite other options in terminal. "
			 "(default: None)")

	group.add_argument("--model", type=str_to_optional_str, default=None,
		help="Model to run. If None, select the default model for the dataset chosen. (default: None)")

	group.add_argument("--dropout", type=float_in_range(0.0, 1.0), default=0.5,
		help="Dropout used in model. WARNING: All models does not use this dropout argument.")

	group.add_argument("--optimizer", "--optim", type=str, default="Adam",
		choices=["Adam", "SGD", "RAdam", "PlainRAdam", "AdamW"],
		help="Optimizer used. (default: Adam)")

	group.add_argument("--learning_rate", "--lr", type=float, default=1e-3,
		help="Learning rate to use. (default: 1e-3)")

	group.add_argument("--weight_decay", "--wd", type=str_to_optional_float, default=None,
		help="Weight decay used. Use None for use the default weight decay of the optimizer. (default: None)")

	group.add_argument("--momentum", type=str_to_optional_float, default=None,
		help="Momentum used in SGD optimizer. Use None for use the default momentum of the optimizer. (default: None)")

	group.add_argument("--use_nesterov", "--nesterov", type=str_to_optional_bool, default=False,
		help="Activate Nesterov momentum for SGD optimizer. Use None for use the default momentum of the optimizer. "
			 "(default: False)")

	group.add_argument("--scheduler", "--sched", type=str_to_optional_str, default=None,
		choices=[
			None,
			"CosineLRScheduler", "cos",
			"MultiStepLR", "step",
			"SoftCosineLRScheduler", "softcos"
		],
		help="FixMatch scheduler used. Use \"None\" for not using any scheduler. (default: None)")

	group.add_argument("--lr_decay_ratio", type=float, default=0.2,
		help="Learning rate decay ratio used in MultiStepLR scheduler. (default: 0.2)")

	group.add_argument("--epoch_steps", type=int, nargs="+", default=[60, 120, 160],
		help="Epochs where we decrease the learning rate. Used in MultiStepLR scheduler. (default: [60, 120, 160])")

	group.add_argument("--checkpoint_path", type=str, default=osp.join("..", "results", "models"),
		help="Directory path where checkpoint models will be saved. (default: \"../results/models\")")

	group.add_argument("--supervised_ratio", "--su_ratio", type=float, default=0.1,
		help="Supervised ratio used for split dataset. (default: 0.1)")

	group.add_argument("--cross_validation", "--cross_val", type=str_to_bool, default=False,
		help="Use cross validation for UBS8K dataset. (default: False)")

	group.add_argument("--fold_val", type=str_to_optional_int, default=None,
		help="Fold used for validation in UBS8K dataset. This parameter is unused if cross validation_old is True. "
							 "(default: None)")

	group.add_argument("--ra_nb_choices", type=int, default=1,
		help="Nb augmentations composed for RandAugment. (default: 1)")

	group.add_argument("--label_smoothing_value", "--smooth", type=float, default=0.0,
		help="Label smoothing value for supervised trainings. Use 0.0 for deactivate label smoothing. "
			 "If value is 1.0, all labels will be a uniform distribution. (default: 0.0)")

	group.add_argument("--batch_size_s", "--bsize_s", "--bsize", type=int, default=128,
		help="Batch size used for supervised loader. (default: 128)")

	group.add_argument("--nb_classes_self_supervised", type=int, default=4,
		help="Nb classes in rotation loss (Self-Supervised part) of ReMixMatch. (default: 4)")

	return parser


def check_args(args: Namespace):
	"""
		Check arguments (mainly directories and files)
		:param args: argparse arguments.
	"""
	if args.write_results:
		if args.logdir is not None and not osp.isdir(args.logdir):
			raise RuntimeError("Invalid dirpath \"{:s}\"".format(args.logdir))
		if args.checkpoint_path is not None and not osp.isdir(args.checkpoint_path):
			raise RuntimeError("Invalid dirpath \"{:s}\"".format(args.checkpoint_path))

	if args.args_filepaths is not None:
		for filepath in args.args_filepaths:
			if not osp.isfile(filepath):
				raise RuntimeError("Invalid filepath \"{:s}\".".format(filepath))


def post_process_args(args: Namespace) -> Namespace:
	"""
		Update arguments by adding some parameters inside.
		:param args: argparse arguments.
		:returns: The updated argparse arguments.
	"""
	if hasattr(args, "args_filepaths"):
		args.git_hash = None
		args_filepaths = args.args_filepaths

		if args.args_filepaths is not None:
			for filepath in args.args_filepaths:
				args = load_args(filepath, args)

		args.args_filepaths = args_filepaths

	args.git_hash = get_current_git_hash()
	return args


def get_current_git_hash() -> str:
	"""
		Return the current git hash in the current directory.
		:returns: The git hash.
	"""
	try:
		git_hash = subprocess.check_output(["git", "describe", "--always"])
		git_hash = git_hash.decode("UTF-8").replace("\n", "")
		return git_hash
	except subprocess.CalledProcessError:
		return "UNKNOWN"


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
				raise RuntimeError("Found unknown(s) key(s) in JSON file : \"{:s}\".".format(", ".join(differences)))

		args.__dict__.update(args_file_dict)

		# Post process : convert "none" strings to None value
		for name, value in args.__dict__.items():
			if isinstance(value, str) and value.lower() == "none":
				args.__dict__[name] = None

	return args
