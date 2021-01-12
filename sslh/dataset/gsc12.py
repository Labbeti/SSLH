
from argparse import Namespace

from mlu.datasets.wrappers import TransformDataset

from sslh.dataset.dataset_sized import DatasetSized
from sslh.dataset.gsc import GSCInterface
from sslh.dataset.detail.gsc_ds import SpeechCommand10
from sslh.dataset.module_call import ModuleCall

from torch.nn import Module
from typing import Callable, Dict, Optional


class GSC12Interface(GSCInterface):
	"""
		Classes: 10 classes + 1 class unknown + 1 class silence
	"""

	def __init__(self):
		super().__init__()
		labels_names_list = [
			"yes", "no", "up", "down", "left", "right", "off", "on", "go", "stop", "silence", "_background_noise_"
		]
		self._labels_names = {index: name for index, name in enumerate(labels_names_list)}
		self._train_percent_to_drop = 0.5

	def get_dataset_train(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = True,
		subset: str = "train",
		**kwargs,
	) -> DatasetSized:
		return _get_dataset(dataset_root, transform, target_transform, download, subset, self._train_percent_to_drop)

	def get_dataset_val(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = True,
		subset: str = "validation",
		**kwargs,
	) -> DatasetSized:
		return _get_dataset(dataset_root, transform, target_transform, download, subset, 0.0)

	def get_dataset_eval(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = True,
		subset: str = "testing",
		**kwargs,
	) -> DatasetSized:
		return _get_dataset(dataset_root, transform, target_transform, download, subset, 0.0)

	def get_dataset_name(self) -> str:
		return "GSC12"

	def get_labels_names(self) -> Dict[int, str]:
		return self._labels_names


def _get_dataset(
	dataset_root: str,
	transform: Optional[Callable],
	target_transform: Optional[Callable],
	download: bool,
	subset: str,
	percent_to_drop: float,
) -> DatasetSized:
	if transform is not None and not isinstance(transform, Module):
		transform = ModuleCall(transform)
	dataset = SpeechCommand10(root=dataset_root, subset=subset, transform=transform, download=download, percent_to_drop=percent_to_drop)
	dataset = TransformDataset(dataset, target_transform, index=1)
	return dataset
