
import numpy as np
import random
import torch

from abc import ABC
from datetime import datetime
from time import time
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataset import Dataset
from typing import Callable, Dict, Generic, List, Optional, TypeVar


def get_lrs(optim: Optimizer) -> List[float]:
	""" Get the learning rates of an optimizer. """
	return [group["lr"] for group in optim.param_groups]


def get_lr(optim: Optimizer, idx: int = 0) -> float:
	""" Get the learning rate of an optimizer. """
	return get_lrs(optim)[idx]


def get_nb_parameters(model: Module) -> int:
	"""
		Return the number of parameters in a model.

		:param model: Pytorch Module to check.
		:returns: The number of parameters.
	"""
	return sum(p.numel() for p in model.parameters())


def get_datetime() -> str:
	"""
		Returns the date in a specific format : "YYYY_MM_DD_hh:mm:ss".
		:returns: The current date.
	"""
	now = str(datetime.now())
	return now[:10] + "_" + now[11:-7]


def reset_seed(seed: int):
	"""
		Reset the seed of following packages : random, numpy, torch, torch.cuda, torch.backends.cudnn.
		:param seed: The seed to set.
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def generate_indexes(
	dataset: Dataset,
	nb_classes: int,
	ratios: List[float],
	shuffle_idx: bool = True,
	target_one_hot: bool = True,
) -> List[List[int]]:
	"""
		Split dataset in list of indexes for each ratio.
		Also keep the original class distribution in every sub-dataset.

		:param dataset: The original dataset.
		:param nb_classes: The number of classes in the original dataset.
		:param ratios: Ratios used to split the dataset. The sum must be 1.
		:param shuffle_idx: Shuffle classes indexes before split them.
		:param target_one_hot: Consider labels as one-hot vectors. If False, consider labels as class indexes.
		:returns: A list of indexes for each ratios.
	"""
	cls_idx_all = _get_classes_idx(dataset, nb_classes, target_one_hot)
	if shuffle_idx:
		cls_idx_all = _shuffle_classes_idx(cls_idx_all)
	indexes = _split_classes_idx(cls_idx_all, ratios)
	return indexes


def _get_classes_idx(dataset: Dataset, nb_classes: int, target_one_hot: bool = True) -> List[List[int]]:
	"""
		Get class indexes from a standard dataset with index of class as label.
	"""
	result = [[] for _ in range(nb_classes)]

	for i in range(len(dataset)):
		_data, label = dataset[i]
		if target_one_hot:
			label = np.argmax(label)
		result[label].append(i)
	return result


def _shuffle_classes_idx(classes_idx: List[List[int]]) -> List[List[int]]:
	"""
		Shuffle each class indexes.
	"""
	result = []
	for indexes in classes_idx:
		random.shuffle(indexes)
		result.append(indexes)
	return result


def _split_classes_idx(classes_idx: List[List[int]], ratios: List[float]) -> List[List[int]]:
	"""
		Split class indexes and merge them for each ratio.

		Ex:
			input:  classes_idx = [[1, 2], [3, 4], [5, 6]], ratios = [0.5, 0.5]
			output: [[1, 3, 5], [2, 4, 6]]
	"""

	result = [[] for _ in range(len(ratios))]

	for indexes in classes_idx:
		current_begin = 0
		for i, ratio in enumerate(ratios):
			current_end = current_begin + int(round(ratio * len(indexes)))
			result[i] += indexes[current_begin:current_end]
			current_begin = current_end
	return result


class ZipDataset(Dataset):
	def __init__(self, datasets: List[Dataset]):
		"""
			Zip through a list of Sized datasets of same sizes.

			:param datasets: The list of dataset to read.
		"""
		super().__init__()
		self.datasets = datasets
		self._check_attributes()

	def _check_attributes(self):
		len_ = len(self.datasets[0])
		for d in self.datasets[1:]:
			assert len(d) == len_, "Datasets must have the same size"

	def __getitem__(self, idx: int) -> list:
		return [d[idx] for d in self.datasets]

	def __len__(self) -> int:
		return len(self.datasets[0]) if len(self.datasets) > 0 else 0


T_Input = TypeVar("T_Input")
T_Target = TypeVar("T_Target")
T_Output = TypeVar("T_Output")


class Metric(Module, Callable, ABC, Generic[T_Input, T_Target, T_Output]):
	"""
		Base class for metric modules.

		Abstract methods:
			- compute_score(self, input_: T_Input, target: T_Target) -> T_Output:
	"""
	def forward(self, input_: T_Input, target: T_Target) -> T_Output:
		return self.compute_score(input_, target)

	def compute_score(self, input_: T_Input, target: T_Target) -> T_Output:
		raise NotImplementedError("Abstract method")


class IncrementalMean:
	def __init__(self):
		"""
			Compute the continue average of a values.
		"""
		super().__init__()
		self._sum = None
		self._counter = 0

	def reset(self):
		self._sum = None
		self._counter = 0

	def add(self, value: Tensor):
		if self._sum is None:
			self._sum = value
			self._counter = 1
		else:
			self._sum += value
			self._counter += 1

	def get_current(self) -> Optional[Tensor]:
		return self.get_mean()

	def is_empty(self) -> bool:
		return self._counter == 0

	def get_mean(self) -> Optional[Tensor]:
		return self._sum / self._counter if self._sum is not None else None


class ColumnPrinter:
	KEY_MAX_LENGTH = 10

	def __init__(self, print_exec_time: bool = True):
		"""
			Class for print current values of a training or validation by columns.

			Ex:
			> printer = ColumnPrinter()
			> printer.print_current_values({"train/accuracy": 0.89, "train/loss": 1.525}, 33, 100, 2)
			-      train       -  accuracy  -    loss    -  took (s)  -
			- Epoch   2 -  33% - 8.9000e-01 - 1.5250e-00 -    0.00    -

			:param print_exec_time: Print time elapsed with the beginning of the loop (iteration == 0).
		"""
		self.print_exec_time = print_exec_time

		self._epoch_start_date = time()
		self._keys = []

	def print_current_values(self, current_values: Dict[str, float], iteration: int, nb_iterations: int, epoch: int):
		if iteration == 0:
			keys = list(sorted(current_values.keys()))
			name = "/".join(keys[0].split("/")[:-1])
			keys_names = [key.split("/")[-1] for key in keys]
			self._keys = keys
			self._print_header(name, keys_names)
			self._epoch_start_date = time()
		else:
			self._keys += list(set(current_values.keys()).difference(self._keys))

		progression = int(100 * (iteration + 1) / nb_iterations)
		content = \
			["Epoch {:3d} - {:3d}%".format(epoch + 1, progression)] + \
			[
				"{:.4e}".format(current_values[key]).center(self.KEY_MAX_LENGTH)
				if key in current_values.keys() else
				" " * self.KEY_MAX_LENGTH
				for key in self._keys
			]

		if self.print_exec_time:
			content += ["{:.2f}".format(time() - self._epoch_start_date).center(self.KEY_MAX_LENGTH)]

		print("- {:s} -".format(" - ".join(content)), end="\r")

		if iteration == nb_iterations - 1:
			print("")

	def _print_header(self, name: str, keys: List[str]):
		def filter_name(key_name: str) -> str:
			if len(key_name) <= self.KEY_MAX_LENGTH:
				return key_name.center(self.KEY_MAX_LENGTH)
			else:
				return key_name[:self.KEY_MAX_LENGTH]

		content = ["{:s}".format(name.center(16))]
		content += [filter_name(metric_name) for metric_name in keys]
		content += ["took (s)".center(self.KEY_MAX_LENGTH)]

		print("- {:s} -".format(" - ".join(content)))


class MaxTracker:
	def __init__(self):
		super().__init__()
		self._max = None
		self._idx_max = -1
		self._index = 0

	def reset(self):
		self._max = None
		self._idx_max = -1
		self._index = 0

	def add(self, value: Tensor):
		if self._max is None or self._max < value:
			self._max = value
			self._idx_max = self._index
		self._index += 1

	def is_empty(self) -> bool:
		return self._max is None

	def get_current(self) -> Optional[Tensor]:
		return self.get_max()

	def get_max(self) -> Optional[Tensor]:
		return self._max

	def get_index(self) -> int:
		return self._idx_max
