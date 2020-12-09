
from torch.utils.data import Dataset

from sslh.datasets.wrappers.transform_dataset import TransformDataset
from sslh.utils.label_conversions import onehot_to_smooth_onehot


class SmoothOneHotDataset(TransformDataset):
	"""
		Convert onehot label to smoothed label.
	"""
	def __init__(self, dataset: Dataset, nb_classes: int, smooth: float):
		convert_label_fn = lambda item: (item[0], onehot_to_smooth_onehot(item[1], nb_classes, smooth))
		super().__init__(dataset, transform=convert_label_fn)
		self.nb_classes = nb_classes
