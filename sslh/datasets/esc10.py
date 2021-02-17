
from mlu.datasets.wrappers import TransformDataset

from sslh.datasets.dataset_sized import DatasetSized
from sslh.datasets.detail.esc_ds import ESC10_NoSR_Cached
from sslh.datasets.esc50 import ESC50Builder
from sslh.datasets.module_wrap import ModuleWrap

from torch.nn import Module
from typing import Callable, Dict, List, Optional


class ESC10Builder(ESC50Builder):
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

		return get_esc10_dataset(
			dataset_root,
			folds,
			download,
			transform,
			target_transform,
		)

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

		return get_esc10_dataset(
			dataset_root,
			folds,
			download,
			transform,
			target_transform,
		)

	def get_dataset_name(self) -> str:
		return "ESC10"

	def get_labels_names(self) -> Dict[int, str]:
		return self._labels_names_esc10


def get_esc10_dataset(
	dataset_root: str,
	folds: List[int],
	download: bool = True,
	transform: Optional[Callable] = None,
	target_transform: Optional[Callable] = None,
) -> DatasetSized:
	if transform is not None and not isinstance(transform, Module):
		transform = ModuleWrap(transform)

	if not isinstance(folds, tuple):
		folds = tuple(folds)

	dataset = ESC10_NoSR_Cached(root=dataset_root, folds=folds, download=download, transform=transform)
	if target_transform is not None:
		dataset = TransformDataset(dataset, transform=target_transform, index=1)
	return dataset
