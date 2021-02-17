
from abc import ABC

from mlu.datasets.utils import generate_indexes

from sslh.datasets.dataset_sized import DatasetSized

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Callable, Dict, List, Optional


class DatasetBuilder(ABC):
	"""
		DatasetBuilder abstract class.

		An builder provides methods for building dataset, augmentations and store specific information for running with standalone scripts.
	"""

	# -------------------------------------------------------------------------
	# Abstract methods
	# -------------------------------------------------------------------------
	def get_dataset_train(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		**kwargs,
	) -> DatasetSized:
		"""
			Build the training dataset.

			:param dataset_root: The path to the dataset root.
			:param transform: The optional transformation to apply to data.
			:param target_transform: The target transformation to labels (targets). (ex: conversion to onehot encoding)
			:param kwargs: The optional arguments for build the dataset class.
			:return: Return the Pytorch dataset class.
		"""
		raise NotImplementedError("Abstract method")

	def get_dataset_val(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		**kwargs,
	) -> DatasetSized:
		"""
			Build the validation dataset.

			:param dataset_root: The path to the dataset root.
			:param transform: The optional transformation to apply to data.
			:param target_transform: The target transformation to labels (targets). (ex: conversion to onehot encoding)
			:param kwargs: The optional arguments for build the dataset class.
			:return: Return the Pytorch dataset class.
		"""
		raise NotImplementedError("Abstract method")

	def get_target_transform(self, **kwargs) -> Optional[Callable]:
		"""
			Return the target transformation. (ex: conversion to onehot encoding)

			:param kwargs: The optional argument for building the transform to apply to labels.
			:return: Return the callable transform. Can return None if no transform must be applied.
		"""
		raise NotImplementedError("Abstract method")

	def get_dataset_name(self) -> str:
		"""
			Return the dataset name or acronym.

			:return: The dataset name.
		"""
		raise NotImplementedError("Abstract method")

	def get_data_type(self) -> str:
		"""
			Return the data type contained in the dataset.

			:return: The data type : "audio" or "image".
		"""
		raise NotImplementedError("Abstract method")

	def get_target_type(self) -> str:
		"""
			Return the type of the dataset labels.

			:return: The label type like "monolabel" or "multilabel".
		"""
		raise NotImplementedError("Abstract method")

	def get_labels_names(self) -> Dict[int, str]:
		"""
			Return the labels names.

			:return: The dictionary of labels. The key must be the corresponding class index.
		"""
		raise NotImplementedError("Abstract method")

	def get_folds(self) -> Optional[List[int]]:
		"""
			Return the number of folds in the dataset.
			Must return None if no folds are available (ie no cross-validation).

			:return: The optional list of folds indexes.
		"""
		raise NotImplementedError("Abstract method")

	def has_evaluation(self) -> bool:
		"""
			:return: Returns True if the evaluation dataset is available.
		"""
		raise NotImplementedError("Abstract method")

	# -------------------------------------------------------------------------
	# Implemented methods (can be override by subclasses)
	# -------------------------------------------------------------------------
	def get_pre_transform(self) -> Optional[Callable]:
		"""
			Return the pre-process transformation.

			:return: Return the transform for data. Can return None if no transform must be applied.
		"""
		return None

	def get_post_transform(self) -> Optional[Callable]:
		"""
			Return the post-process transformation.

			:return: Return the transform for data. Can return None if no transform must be applied.
		"""
		return None

	def get_spec_transform(self) -> Optional[Callable]:
		"""
			Return the transformation to spectrogram.

			:return: Return the transform to spectrogram for data. Can return None if no transform must be applied.
		"""
		return None

	def get_dataset_eval(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		**kwargs,
	) -> Optional[DatasetSized]:
		"""
			Build the evaluation dataset.

			:param dataset_root: The path to the dataset root.
			:param transform: The optional transformation to apply to data.
			:param target_transform: The target transformation to labels (targets). (ex: conversion to onehot encoding)
			:param kwargs: The optional arguments for build the dataset class.
			:return: Return the Pytorch dataset class if an evaluation dataset is available.
		"""
		return None

	def has_folds(self) -> bool:
		"""
			:return: Returns True if dataset is preprocessed for cross-validation.
		"""
		return self.get_folds() is not None

	def get_nb_classes(self) -> int:
		"""
			:return: The number of classes in the dataset.
		"""
		return len(self.get_labels_names())

	def get_loader_train(
		self,
		dataset: DatasetSized,
		batch_size: int,
		drop_last: bool = False,
		num_workers: int = 1,
		shuffle: bool = True,
		pin_memory: bool = False,
		**kwargs,
	) -> DataLoader:
		"""
			Build the train dataloader with a dataset.

			:param dataset: THe dataset to load.
			:param batch_size: The batch_size argument of the dataloader.
			:param drop_last: The drop_last argument of the dataloader.
			:param num_workers: The num_workers argument of the dataloader.
			:param shuffle: The shuffle argument of the dataloader.
			:param pin_memory: The pin_memory argument of the dataloader.
			:param kwargs: Unused.
			:return: The DataLoader built.
		"""
		return DataLoader(
			dataset,
			batch_size=batch_size,
			drop_last=drop_last,
			num_workers=num_workers,
			shuffle=shuffle,
			pin_memory=pin_memory,
		)

	def get_loader_val(
		self,
		dataset: DatasetSized,
		batch_size: int,
		drop_last: bool = True,
		num_workers: int = 0,
		shuffle: bool = False,
		pin_memory: bool = False,
		**kwargs,
	) -> DataLoader:
		"""
			Build the validation dataloader with a dataset.

			:param dataset: The dataset to load.
			:param batch_size: The batch_size argument of the dataloader.
			:param drop_last: The drop_last argument of the dataloader.
			:param num_workers: The num_workers argument of the dataloader.
			:param shuffle: The shuffle argument of the dataloader.
			:param pin_memory: The pin_memory argument of the dataloader.
			:param kwargs: Unused.
			:return: The DataLoader built.
		"""
		return DataLoader(
			dataset,
			batch_size=batch_size,
			drop_last=drop_last,
			num_workers=num_workers,
			shuffle=shuffle,
			pin_memory=pin_memory,
		)

	def get_loaders_split(
		self,
		labeled_dataset: DatasetSized,
		ratios: List[float],
		datasets: List[DatasetSized],
		batch_sizes: List[int],
		drop_last_list: List[bool],
		num_workers_list: List[int],
		target_transformed: bool = False,
		**kwargs,
	) -> List[DataLoader]:
		"""
			Split the datasets and build the corresponding dataloaders.

			:param labeled_dataset: The complete labeled dataset used for create a split that keeps the classes distributions.
			:param ratios: The ratios of each split. Can be constraint by each dataset builder.
			:param datasets: The datasets to load.
			:param batch_sizes: The batch_size arguments of the dataloader.
			:param drop_last_list: The drop_last arguments of the dataloader.
			:param num_workers_list: The num_workers arguments of the dataloader.
			:param target_transformed:
				If True, the target of the labeled dataset will be considered as transformed by the target transform of the dataset.
			:param kwargs: Unused.
			:return: The DataLoader built.
		"""
		assert len(ratios) == len(datasets) == len(batch_sizes) == len(drop_last_list) == len(num_workers_list)

		if self.get_target_type() == "monolabel":
			indexes = generate_indexes(labeled_dataset, self.get_nb_classes(), ratios, target_one_hot=target_transformed)
			samplers = [SubsetRandomSampler(idx) for idx in indexes]

			loaders = [
				DataLoader(
					dataset=dataset,
					batch_size=batch_size,
					drop_last=drop_last,
					num_workers=num_workers,
					sampler=sampler,
				)
				for dataset, batch_size, drop_last, num_workers, sampler
				in zip(datasets, batch_sizes, drop_last_list, num_workers_list, samplers)
			]

			return loaders
		else:
			raise NotImplementedError("No default split method for an non-monolabel dataset.")
