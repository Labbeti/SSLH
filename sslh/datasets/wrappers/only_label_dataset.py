
from sslh.datasets.wrappers.transform_dataset import TransformDataset
from torch.utils.data import Dataset


class OnlyLabelDataset(TransformDataset):
	"""
		Wrapper of Dataset which remove data from dataset by getting only the batch.
	"""
	def __init__(self, dataset: Dataset, label_idx: int = 1):
		super().__init__(dataset, transform=lambda item: item[label_idx])
