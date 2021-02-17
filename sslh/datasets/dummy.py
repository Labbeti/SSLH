from typing import Optional, List, Dict, Callable

from datasets.dataset_sized import DatasetSized
from sslh.datasets.base import DatasetBuilder
from torch.utils.data.dataset import Dataset


class DummyDataset(Dataset):
	def __init__(self, size: int, transform: Optional[Callable], target_transform: Optional[Callable]):
		super().__init__()
		self.size = size
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index: int) -> tuple:
		return self.transform(index), self.target_transform(index % 10)


class DummyBuilder(DatasetBuilder):
	def get_dataset_train(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		**kwargs,
	) -> DatasetSized:
		return DummyDataset(100, transform, target_transform)

	def get_dataset_val(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		**kwargs,
	) -> DatasetSized:
		return DummyDataset(100, transform, target_transform)

	def get_target_transform(self, **kwargs) -> Optional[Callable]:
		return None

	def get_dataset_name(self) -> str:
		return "DUMMY"

	def get_data_type(self) -> str:
		return "audio"

	def get_target_type(self) -> str:
		return "monolabel"

	def get_labels_names(self) -> Dict[int, str]:
		return {i: str(i) for i in range(10)}

	def get_folds(self) -> Optional[List[int]]:
		return None

	def has_evaluation(self) -> bool:
		return False
