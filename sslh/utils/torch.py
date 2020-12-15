
import torch

from torch import Tensor
from typing import List


def normalized(batch: Tensor, dim: int) -> Tensor:
	"""
		Return the vector normalized.
	"""
	return batch / batch.norm(p=1, dim=dim, keepdim=True)


def same_shuffle(values: List[Tensor]) -> List[Tensor]:
	"""
		Shuffle each value of values with the same indexes.
	"""
	indices = torch.randperm(len(values[0]))
	for i in range(len(values)):
		values[i] = values[i][indices]
	return values


def merge_first_dimension(t: Tensor) -> Tensor:
	"""
		Reshape tensor of size (M, N, ...) to (M*N, ...).
	"""
	shape = list(t.shape)
	if len(shape) < 2:
		raise RuntimeError("Invalid nb of dimension ({:d}) for merge_first_dimension. Should have at least 2 dimensions.".format(len(shape)))
	return t.reshape(shape[0] * shape[1], *shape[2:])
