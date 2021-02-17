
import torch

from torch import Tensor
from typing import List


def normalized(batch: Tensor, dim: int) -> Tensor:
	"""
		Return the vector normalized.
	"""
	return batch / batch.norm(p=1, dim=dim, keepdim=True)


def collapse_first_dimension(t: Tensor) -> Tensor:
	"""
		Reshape tensor of shape (M, N, ...) to (M*N, ...).
	"""
	shape = list(t.shape)
	if len(shape) < 2:
		raise RuntimeError(
			f"Invalid nb of dimension ({len(shape)}) for merge_first_dimension. Should have at least 2 dimensions.")
	return t.reshape(shape[0] * shape[1], *shape[2:])
