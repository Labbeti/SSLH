
from sslh.utils.label_conversions import nums_to_onehot, nums_to_smooth_onehot
from torch.utils.data import Dataset
from typing import Optional, Sized


class OneHotDataset(Dataset, Sized):
	"""
		Dataset wrapper for convert labels to one-hot.
	"""
	def __init__(self, dataset: Dataset, nb_classes: int, smooth: Optional[float] = None):
		super().__init__()
		self.dataset = dataset
		if smooth is None or smooth == 0.0:
			self.convert_fn = lambda nums: nums_to_onehot(nums, nb_classes)
		else:
			self.convert_fn = lambda nums: nums_to_smooth_onehot(nums, nb_classes, smooth)

	def __getitem__(self, idx: int) -> tuple:
		item = self.dataset.__getitem__(idx)
		return item[0], self.convert_fn(item[1])

	def __len__(self) -> int:
		return len(self.dataset)
