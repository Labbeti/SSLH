import os.path as osp

from argparse import Namespace

from mlu.datasets.wrappers import OneHotDataset
from mlu.transforms import Identity

from sslh.augments.get_pool import get_pool_audio_with_name
from sslh.datasets.abc import DatasetInterface
from sslh.datasets.dataset_sized import DatasetSized
from sslh.datasets.transform import get_transform_self_supervised_flips
from sslh.models.cnn03 import CNN03, CNN03Rot
from sslh.models.ubs8k_baseline import UBS8KBaseline, UBS8KBaselineRot
from sslh.models.wrn28_2 import WideResNet28Spec, WideResNet28RotSpec

from torchvision.transforms import RandomChoice
from typing import Callable, List, Optional

from ubs8k.datasets import Dataset as UBS8KDataset
from ubs8k.datasetManager import DatasetManager as UBS8KDatasetManager


class UBS8KInterface(DatasetInterface):
	def __init__(self):
		super().__init__()
		self._manager = None

	def get_dataset_train_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		if folds is None:
			folds = (1, 2, 3, 4, 5, 6, 7, 8, 9)

		if transform is None:
			transform = ()
		elif isinstance(transform, RandomChoice):
			transform = tuple(transform.transforms)
		else:
			transform = (transform,)

		manager = self._load_manager(args.dataset_path)
		dataset_train = UBS8KDataset(manager, folds=folds, augments=transform, cached=False, augment_choser=lambda x: x)
		dataset_train = OneHotDataset(dataset_train, self.get_nb_classes(), args.label_smoothing_value)
		return dataset_train

	def get_dataset_val_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		if folds is None:
			folds = (10,)

		if transform is None:
			transform = ()
		elif isinstance(transform, RandomChoice):
			transform = tuple(transform.transforms)
		else:
			transform = (transform,)

		manager = self._load_manager(args.dataset_path)
		dataset_val = UBS8KDataset(manager, folds=folds, augments=transform, cached=True, augment_choser=lambda x: x)
		dataset_val = OneHotDataset(dataset_val, self.get_nb_classes())
		return dataset_val

	def get_dataset_eval_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> Optional[DatasetSized]:
		return None

	def get_transform_augm_none(self, args: Optional[Namespace]) -> Callable:
		pool_none = get_pool_audio_with_name(args.augm_none, args)
		return RandomChoice(pool_none)

	def get_transform_augm_weak(self, args: Optional[Namespace]) -> Callable:
		pool_weak = get_pool_audio_with_name(args.augm_weak, args)
		return RandomChoice(pool_weak)

	def get_transform_augm_strong(self, args: Optional[Namespace]) -> Callable:
		pool_strong = get_pool_audio_with_name(args.augm_strong, args)
		return RandomChoice(pool_strong)

	def get_transform_val(self, args: Optional[Namespace]) -> Callable:
		return Identity()

	def get_dataset_name(self) -> str:
		return "UBS8K"

	def get_nb_classes(self) -> int:
		return 10

	def get_nb_folds(self) -> Optional[int]:
		return 10

	def get_default_model_name(self) -> str:
		return "WideResNet28Spec"

	def get_models(self) -> list:
		return [CNN03, CNN03Rot, UBS8KBaseline, UBS8KBaselineRot, WideResNet28Spec, WideResNet28RotSpec]

	def get_transform_self_supervised(self, args: Optional[Namespace] = None) -> Callable:
		return get_transform_self_supervised_flips(args)

	def get_class_name(self, index: int) -> str:
		raise NotImplementedError("TODO")

	def _load_manager(self, dataset_path: str) -> UBS8KDatasetManager:
		if self._manager is None:
			metadata_root = osp.join(dataset_path, "metadata")
			audio_root = osp.join(dataset_path, "audio")
			self._manager = UBS8KDatasetManager(metadata_root, audio_root)
		return self._manager
