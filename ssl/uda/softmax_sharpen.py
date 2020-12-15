
from ssl.utils.torch import normalized

from torch import Tensor
from torch.nn import Module
from typing import Optional


def softmax_sharpen(batch: Tensor, temperature: float, dim: int) -> Tensor:
	batch = batch / temperature
	batch = batch.exp()
	return normalized(batch, dim)


class SoftmaxSharpen(Module):
	"""
		Softmax sharpening class used in UDA.
	"""
	def __init__(self, temperature: float = 0.4, dim: Optional[int] = 1):
		super().__init__()
		self.temperature = temperature
		self.dim = dim

	def forward(self, x: Tensor, dim: Optional[int] = None) -> Tensor:
		if dim is None:
			dim = self.dim
		if dim is None:
			raise RuntimeError("None dimension found. Dimension is required in forward(x, dim) or in __init__ of the class.")
		return softmax_sharpen(x, self.temperature, dim)
