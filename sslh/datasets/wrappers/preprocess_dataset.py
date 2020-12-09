
from torch.utils.data import Dataset
from typing import Callable

from sslh.datasets.wrappers.transform_dataset import TransformDataset


class PreProcessDataset(TransformDataset):
	"""
		Apply augment function on first element of every items get in dataset.
	"""
	def __init__(self, dataset: Dataset, preprocess_fn: Callable, has_label: bool = True):
		fn_label = lambda item: (preprocess_fn(item[0]),) + tuple(item[1:])
		fn_no_label = lambda item: preprocess_fn(item)
		fn = fn_label if has_label else fn_no_label
		super().__init__(dataset, transform=fn)
