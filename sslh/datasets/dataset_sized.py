
from torch.utils.data import Dataset
from typing import Sized, Protocol, Union


class DatasetSizedProtocol(Sized, Protocol):
	"""
		Base class which implements a Dataset with a __len__ method.
	"""
	def __getitem__(self, index):
		raise NotImplementedError("Abstract method")

	def __add__(self, other):
		raise NotImplementedError("Abstract method")

	def __len__(self) -> int:
		raise NotImplementedError("Abstract method")


DatasetSized = Union[Dataset, DatasetSizedProtocol]
