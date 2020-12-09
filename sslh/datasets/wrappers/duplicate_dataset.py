
from torch.utils.data import Dataset


class DuplicateDataset(Dataset):
	"""
		Concatenate items of wrappers of the same size.
	"""
	def __init__(self, dataset: Dataset, nb_duplicates: int):
		super(DuplicateDataset, self).__init__()
		self.dataset = dataset
		self.nb_duplicates = nb_duplicates

	def __getitem__(self, idx: int) -> list:
		item = self.dataset[idx]
		return [item] * self.nb_duplicates

	def __len__(self) -> int:
		return len(self.dataset)
