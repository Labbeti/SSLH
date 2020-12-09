import torch

from abc import ABC
from argparse import Namespace
from sslh.datasets.utils import get_classes_idx, shuffle_classes_idx, split_classes_idx
from sslh.datasets.dataset_sized import DatasetSized
from sslh.models.get_model import get_model

from torch.nn import Module
from torch.utils.data import Dataset
from typing import Callable, Dict, Optional, List


class DatasetInterface(ABC):
	"""
		DatasetInterface abstract class.

		An interface provide method for building dataset, augmentations and models specific for running with standalone scripts.
	"""

	# Abstract methods
	def get_dataset_train_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		"""
			Return the train dataset with specifics folds and transform.
		"""
		raise NotImplementedError("Abstract method")

	def get_dataset_val_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		"""
			Return the validation dataset with specifics folds and transform.
		"""
		raise NotImplementedError("Abstract method")

	def get_dataset_eval_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> Optional[DatasetSized]:
		"""
			Return the evaluation dataset with specific folds.
		"""
		raise NotImplementedError("Abstract method")

	def get_transform_augm_none(self, args: Optional[Namespace]) -> Callable:
		"""
			Return the transformation to use when no augmentation is applied on train dataset.
		"""
		raise NotImplementedError("Abstract method")

	def get_transform_augm_weak(self, args: Optional[Namespace]) -> Callable:
		"""
			Return the transformation to use when the weak augmentation is applied on train dataset.
		"""
		raise NotImplementedError("Abstract method")

	def get_transform_augm_strong(self, args: Optional[Namespace]) -> Callable:
		"""
			Return the transformation to use when the strong augmentation is applied on train dataset.
		"""
		raise NotImplementedError("Abstract method")

	def get_transform_val(self, args: Optional[Namespace]) -> Callable:
		"""
			Return the transformation to use when the no augmentation is applied on validation dataset.
		"""
		raise NotImplementedError("Abstract method")

	def get_dataset_name(self) -> str:
		"""
			Return the dataset name or acronym.
		"""
		raise NotImplementedError("Abstract method")

	def get_nb_classes(self) -> int:
		"""
			Return the number of classes in the dataset.
		"""
		raise NotImplementedError("Abstract method")

	def get_nb_folds(self) -> Optional[int]:
		"""
			Return the number of folds in the dataset. Must return None if no folds are available (ie no cross-validation).
		"""
		raise NotImplementedError("Abstract method")

	def get_default_model_name(self) -> str:
		"""
			Return the name of the default model used to train on the dataset.
		"""
		raise NotImplementedError("Abstract method")

	def get_models(self) -> list:
		"""
			Return the list of model classes available for training on this dataset.
		"""
		raise NotImplementedError("Abstract method")

	def get_transform_self_supervised(self, args: Optional[Namespace] = None) -> Callable:
		"""
			Return the self-supervised transform used in ReMixMatch for this dataset.
		"""
		raise NotImplementedError("Abstract method")

	def get_class_name(self, index: int) -> str:
		raise NotImplementedError("Abstract method")

	# Implemented methods
	def build_model(self, model_name: Optional[str], args: Optional[Namespace], device: torch.device = torch.device("cuda")) -> Module:
		model_name = model_name if model_name is not None else self.get_default_model_name()
		models_classes = self.get_models()
		model = get_model(model_name, args, models_classes, device)
		return model

	def get_indexes(self, dataset: Dataset, ratios: List[float]) -> List[List[int]]:
		cls_idx_all = get_classes_idx(dataset, self.get_nb_classes(), is_one_hot=True)
		cls_idx_all = shuffle_classes_idx(cls_idx_all)
		idx_split = split_classes_idx(cls_idx_all, ratios)
		return idx_split

	def get_dataset_train(self, args: Namespace, folds: Optional[List[int]] = None) -> DatasetSized:
		transform_augm_none = self.get_transform_augm_none(args)
		return self.get_dataset_train_with_transform(args, folds, transform_augm_none)

	def get_dataset_train_augm_weak(self, args: Namespace, folds: Optional[List[int]] = None) -> DatasetSized:
		transform_augm_weak = self.get_transform_augm_weak(args)
		return self.get_dataset_train_with_transform(args, folds, transform_augm_weak)

	def get_dataset_train_augm_strong(self, args: Namespace, folds: Optional[List[int]] = None) -> DatasetSized:
		transform_augm_strong = self.get_transform_augm_strong(args)
		return self.get_dataset_train_with_transform(args, folds, transform_augm_strong)

	def get_dataset_val(self, args: Namespace, folds: Optional[List[int]] = None) -> DatasetSized:
		transform_val = self.get_transform_val(args)
		return self.get_dataset_val_with_transform(args, folds, transform_val)

	def get_dataset_eval(self, args: Namespace, folds: Optional[List[int]] = None) -> DatasetSized:
		transform_val = self.get_transform_eval(args)
		return self.get_dataset_eval_with_transform(args, folds, transform_val)

	def get_transforms(self, args: Optional[Namespace]) -> Dict[str, Callable]:
		transforms = {}
		if args is not None:
			if hasattr(args, "augm_none"):
				transforms["augm_none"] = self.get_transform_augm_none(args)
			if hasattr(args, "augm_weak"):
				transforms["augm_weak"] = self.get_transform_augm_weak(args)
			if hasattr(args, "augm_strong"):
				transforms["augm_strong"] = self.get_transform_augm_strong(args)
		return transforms

	def get_transform_eval(self, args: Optional[Namespace]) -> Callable:
		return self.get_transform_val(args)
