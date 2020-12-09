
from torch.utils.data import Dataset
from typing import Callable


class TransformDataset(Dataset):
	""" Apply a function (func) on before getting items. """
	def __init__(self, dataset: Dataset, transform: Callable):
		super().__init__()
		self.dataset = dataset
		self.transform = transform

	def __getitem__(self, idx: int):
		return self.transform(self.dataset.__getitem__(idx))

	def __len__(self) -> int:
		if hasattr(self.dataset, "__len__"):
			return self.dataset.__len__()
		else:
			raise RuntimeError("The dataset is not sized.")
