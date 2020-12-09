
from sslh.datasets.wrappers.transform_dataset import TransformDataset
from torch.utils.data import Dataset


class NoLabelDataset(TransformDataset):
	"""
		Wrapper of Dataset which remove label from dataset by getting only the batch.
	"""
	def __init__(self, dataset: Dataset, data_idx: int = 0):
		super().__init__(dataset, transform=lambda item: item[data_idx])
