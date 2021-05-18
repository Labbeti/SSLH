
from torch.nn import Module
from typing import Any, Callable, Tuple


class MixMatchUnlabeledPreProcess(Module):
	"""
		Compose transform_weak for unlabeled data.

		Note: (weak(data), weak(data), ...)
	"""
	def __init__(self, transform_weak: Callable, n_augms: int):
		super().__init__()
		self.transform_weak = transform_weak
		self.n_augms = n_augms

	def forward(self, data: Any) -> Tuple[Any, ...]:
		return tuple(self.transform_weak(data) for _ in range(self.n_augms))
