
from torch.utils.data import Dataset
from typing import List


class MultipleDataset(Dataset):
	"""
		Concatenate items of wrappers of the same size.
	"""
	def __init__(self, datasets: List[Dataset]):
		super(MultipleDataset, self).__init__()
		self.datasets = datasets

		self._check_attributes()

	def _check_attributes(self):
		assert len(self.datasets) > 0, "Datasets should not be an empty list"

		len_ = len(self.datasets[0])
		for d in self.datasets[1:]:
			assert len(d) == len_, "Datasets must have the same size"

	def __len__(self) -> int:
		return len(self.datasets[0])

	def __getitem__(self, idx: int) -> list:
		return [d[idx] for d in self.datasets]
