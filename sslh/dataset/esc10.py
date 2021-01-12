
from mlu.datasets.wrappers import TransformDataset

from sslh.dataset.dataset_sized import DatasetSized
from sslh.dataset.detail.esc_ds import ESC10_NoSR_Cached
from sslh.dataset.esc50 import ESC50Interface
from sslh.dataset.module_call import ModuleCall

from torch.nn import Module
from typing import Callable, Dict, List, Optional


class ESC10Interface(ESC50Interface):
	def __init__(self):
		super().__init__()
		self._esc10_idx_to_esc50_idx = {v: k for k, v in ESC10_NoSR_Cached.TARGET_MAPPER.items()}
		self._labels_names_esc10 = {
			index: self._labels_names[self._esc10_idx_to_esc50_idx[index]]
			for index in range(len(self._esc10_idx_to_esc50_idx))
		}

	def get_dataset_train(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		folds: Optional[List[int]] = None,
		download: bool = True,
		**kwargs,
	) -> DatasetSized:
		if folds is None:
			folds = self.get_folds()
			folds = folds[:-1]

		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)

		dataset = ESC10_NoSR_Cached(root=dataset_root, folds=folds, download=download, transform=transform)
		dataset = TransformDataset(dataset, transform=target_transform, index=1)
		return dataset

	def get_dataset_val(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		folds: Optional[List[int]] = None,
		download: bool = True,
		**kwargs,
	) -> DatasetSized:
		if folds is None:
			folds = self.get_folds()
			folds = folds[-1:]

		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)

		dataset = ESC10_NoSR_Cached(root=dataset_root, folds=folds, download=download, transform=transform)
		dataset = TransformDataset(dataset, transform=target_transform, index=1)
		return dataset

	def get_dataset_name(self) -> str:
		return "ESC10"

	def get_labels_names(self) -> Dict[int, str]:
		return self._labels_names_esc10
