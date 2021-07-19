"""
	Primate Vocalisations Corpus (PVC) core classes and functions.
	Developed by Léo Cances (leocances on Github).

	Modified : Yes
		- typing & imports
"""

import functools
import itertools
import numpy
import random

from tqdm import tqdm
from torch.nn import Module
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler
from typing import List, Tuple

from sslh.datasets.pvc_base import COMPARE2021PRSBase
from sslh.datasets.utils import cache_feature


class ComParE2021PRS(COMPARE2021PRSBase):
	def __init__(self, root, subset, transform: Module = None, enable_cache: bool = False):
		super().__init__(root, subset)
		self.transform = transform
		self.enable_cache = enable_cache

	@cache_feature
	def __getitem__(self, idx: int):
		data, target = super().__getitem__(idx)

		if self.transform is not None:
			data = self.transform(data)

		return data, target


def class_balance_split(
	dataset,
	supervised_ratio: float = 0.1,
	unsupervised_ratio: float = None,
	verbose: bool = False,
):
	def to_one_hot(idx):
		one_hot = [0] * len(COMPARE2021PRSBase.CLASSES)
		one_hot[idx] = 1

		return one_hot

	def fill_subset(remaining_samples, expected):
		n_classes = len(COMPARE2021PRSBase.CLASSES)

		subset_occur = numpy.zeros(shape=(n_classes,))
		subset = []

		with tqdm(total=sum(expected)) as progress:
			for class_idx in range(n_classes):
				idx = 0
				while idx < len(remaining_samples) and subset_occur[class_idx] < expected[class_idx]:
					if remaining_samples[idx][0][class_idx] == 1:
						target, target_idx = remaining_samples.pop(idx)
						subset_occur += target
						subset.append(target_idx)
						progress.update(sum(target))

					idx += 1

		return numpy.asarray(subset), remaining_samples

	if unsupervised_ratio is None:
		unsupervised_ratio = 1 - supervised_ratio

	assert 0.0 <= supervised_ratio <= 1.0
	assert 0.0 <= unsupervised_ratio <= 1.0
	assert supervised_ratio + unsupervised_ratio <= 1.0

	if supervised_ratio == 1.0:
		return list(range(len(dataset))), []

	all_targets = list(map(to_one_hot, dataset.subsets_info['target']))
	all_target_idx = list(range(len(all_targets)))

	# expected occurrence and tolerance
	total_occur = numpy.sum(all_targets, axis=0)
	s_expected_occur = numpy.ceil(total_occur * supervised_ratio)
	# u_expected_occur = numpy.ceil(total_occur * unsupervised_ratio)

	if verbose:
		print('s_expected_occur: ', s_expected_occur)
		print('sum s expected occur: ', sum(s_expected_occur))

	all_sample = list(zip(all_targets, all_target_idx))
	s_subset, remaining_sample = fill_subset(all_sample, s_expected_occur)
	u_subset = numpy.asarray([s[1] for s in remaining_sample])

	return s_subset, u_subset


class IterationBalancedSampler(Sampler):
	def __init__(self, dataset: ComParE2021PRS, index_list: List[int], n_max_samples: int, shuffle: bool = True):
		super().__init__(None)
		self.dataset = dataset
		self.index_list = index_list
		self.n_max_samples = n_max_samples
		self.shuffle = shuffle

		self.all_targets = dataset.subsets_info['target']
		self.sorted_sample_indexes = self._sort_per_class()

	def _sort_per_class(self) -> List[List[int]]:
		n_classes = len(COMPARE2021PRSBase.CLASSES)

		class_indexes = [[] for _ in range(n_classes)]
		class_indexes: List[List[int]]

		for sample_idx, target in zip(self.index_list, self.all_targets):
			target_idx = [target]

			for t_idx in target_idx:
				class_indexes[t_idx].append(sample_idx)

		return class_indexes

	def _shuffle(self):
		# Sort the file for each class
		for i in self.sorted_sample_indexes:
			random.shuffle(i)

		# Sort the class order
		random.shuffle(self.sorted_sample_indexes)

	def __len__(self) -> int:
		return self.n_max_samples

	def __iter__(self):
		""" Round Robin algorithm to fetch file one by one from each class.
		"""
		if self.shuffle:
			self._shuffle()

		n_classes = len(COMPARE2021PRSBase.CLASSES)

		global_index = 0
		for cls_idx in itertools.cycle(range(n_classes)):
			# Increment the global index everytime we looped through all the classes
			if cls_idx == 0:
				global_index += 1

			selected_class = self.sorted_sample_indexes[cls_idx]
			local_idx = global_index % len(selected_class)

			yield selected_class[local_idx]


class InfiniteSampler(Sampler):
	def __init__(self, index_list: list, shuffle: bool = True):
		super().__init__(None)
		self.index_list = index_list

	def _shuffle(self):
		random.shuffle(self.index_list)

	@functools.lru_cache(maxsize=1)
	def __len__(self):
		return len(self.index_list)

	def __iter__(self):
		for i, idx in enumerate(itertools.cycle(self.index_list)):
			if i % len(self) == 0:
				self._shuffle()

			yield idx


def mean_teacher(
		root,
		supervised_ratio: float = 0.1,
		batch_size: int = 128,

		train_transform: Module = None,
		val_transform: Module = None,

		n_workers: int = 4,
		pin_memory: bool = False,
		seed: int = 1234,

		**kwargs) -> Tuple[DataLoader, DataLoader]:
	loader_args = {
		'n_workers': n_workers,
		'pin_memory': pin_memory,
	}

	# Training subset
	train_dataset = ComParE2021PRS(root=root, subset='train', transform=train_transform)
	print(train_dataset.__getitem__)
	s_idx, u_idx = class_balance_split(train_dataset, supervised_ratio)

	s_batch_size = int(numpy.floor(batch_size * supervised_ratio))
	u_batch_size = int(numpy.ceil(batch_size * (1 - supervised_ratio)))

	print('s_idx: ', len(s_idx))
	print('u_idx: ', len(u_idx))

	sampler_s = IterationBalancedSampler(train_dataset, s_idx, len(s_idx), shuffle=True)
	#     sampler_s = SubsetRandomSampler(s_idx)
	sampler_u = InfiniteSampler(u_idx, shuffle=True)

	train_s_loader = DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s, **loader_args)
	train_u_loader = DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u, **loader_args)

	# train_loader = ZipCycleInfinite([train_s_loader, train_u_loader])
	train_loader = train_s_loader

	# validation subset
	val_dataset = ComParE2021PRS(root=root, subset='devel', transform=val_transform)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_args)

	return train_loader, val_loader
