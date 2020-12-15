
import numpy as np
import random

from sslh.datasets.dataset_sized import DatasetSized
from torch.utils.data import Dataset, Subset
from typing import List

DEFAULT_EPSILON = 2e-20


def get_classes_idx(dataset: DatasetSized, nb_classes: int, is_one_hot: bool = True) -> List[List[int]]:
	"""
		Get class indexes from a standard dataset with index of class as label.
	"""
	result = [[] for _ in range(nb_classes)]

	for i in range(len(dataset)):
		_data, label = dataset[i]
		if is_one_hot:
			label = np.argmax(label)
		result[label].append(i)
	return result


def shuffle_classes_idx(classes_idx: List[List[int]]) -> List[List[int]]:
	"""
		Shuffle each class indexes.
	"""
	result = []
	for indexes in classes_idx:
		random.shuffle(indexes)
		result.append(indexes)
	return result


def reduce_classes_idx(classes_idx: List[List[int]], ratio: float) -> List[List[int]]:
	"""
		Reduce class indexes by a ratio.
	"""
	result = []
	for indexes in classes_idx:
		idx_dataset_end = max(int(len(indexes) * ratio), 0)
		indexes = indexes[:idx_dataset_end]
		result.append(indexes)
	return result


def split_classes_idx(classes_idx: List[List[int]], ratios: List[float]) -> List[List[int]]:
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


def split_dataset(
	dataset: Dataset,
	nb_classes: int,
	ratios: List[float],
	shuffle_idx: bool = True,
	is_one_hot: bool = True,
) -> List[Dataset]:
	"""
		Split dataset in several sub-wrappers by using a list of ratios.
		Also keep the original class distribution in every sub-dataset.
		:param dataset: The original dataset.
		:param nb_classes: The number of classes in the original dataset.
		:param ratios: Ratios used to split the dataset. The sum must be 1.
		:param shuffle_idx: Shuffle classes indexes before split them.
		:param is_one_hot: Consider labels as one-hot vectors. If False, consider labels as class indexes.
		:returns: A list of sub-wrappers.
	"""
	cls_idx_all = get_classes_idx(dataset, nb_classes, is_one_hot)
	if shuffle_idx:
		cls_idx_all = shuffle_classes_idx(cls_idx_all)
	idx_split = split_classes_idx(cls_idx_all, ratios)

	return [Subset(dataset, idx) for idx in idx_split]
