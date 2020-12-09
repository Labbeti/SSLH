
from torch.utils.data import Dataset, Subset
from sslh.datasets.utils import get_classes_idx, shuffle_classes_idx, reduce_classes_idx, \
	collapse_classes_idx


class ReducedDataset(Dataset):
	def __init__(self, dataset: Dataset, nb_classes: int, ratio: float):
		cls_idx_all = get_classes_idx(dataset, nb_classes)
		cls_idx_all = shuffle_classes_idx(cls_idx_all)
		cls_idx_all = reduce_classes_idx(cls_idx_all, ratio)
		indexes = collapse_classes_idx(cls_idx_all)
		self.dataset = Subset(dataset, indexes)

	def __getitem__(self, idx: int):
		return self.dataset.__getitem__(idx)

	def __len__(self) -> int:
		return len(self.dataset)
