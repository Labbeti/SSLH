
from argparse import Namespace

from sslh.datasets.dataset_sized import DatasetSized
from sslh.datasets.detail.esc_ds import ESC10_NoSR_Cached
from sslh.datasets.esc50 import ESC50Interface
from sslh.datasets.module_call import ModuleCall
from mlu.datasets.wrappers import OneHotDataset

from torch.nn import Module
from typing import Callable, Optional, List


class ESC10Interface(ESC50Interface):
	def __init__(self):
		super().__init__()
		self._esc10_idx_to_esc50_idx = {v: k for k, v in ESC10_NoSR_Cached.TARGET_MAPPER.items()}

	def get_dataset_train_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		if folds is None:
			folds = (1, 2, 3, 4)
		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)

		dataset_train = ESC10_NoSR_Cached(root=args.dataset_path, folds=folds, download=True, transform=transform)
		dataset_train = OneHotDataset(dataset_train, self.get_nb_classes(), smooth=args.label_smoothing_value)
		return dataset_train

	def get_dataset_val_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		if folds is None:
			folds = (5,)
		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)

		dataset_val = ESC10_NoSR_Cached(root=args.dataset_path, folds=folds, download=True, transform=transform)
		dataset_val = OneHotDataset(dataset_val, self.get_nb_classes(), smooth=None)
		return dataset_val

	def get_dataset_name(self) -> str:
		return "ESC10"

	def get_nb_classes(self) -> int:
		return 10

	def get_class_name(self, index: int) -> str:
		esc50_idx = self._esc10_idx_to_esc50_idx[index]
		return super().get_class_name(esc50_idx)
