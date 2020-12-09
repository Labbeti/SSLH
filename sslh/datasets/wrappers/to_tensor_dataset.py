import torch

from torch.utils.data import Dataset
from sslh.datasets.wrappers.transform_dataset import TransformDataset


class ToTensorDataset(TransformDataset):
	"""
		Concert items to Tensors.
	"""
	def __init__(self, dataset: Dataset):
		super().__init__(dataset, transform=lambda item: tuple([torch.as_tensor(elt.tolist()) for elt in item]))
