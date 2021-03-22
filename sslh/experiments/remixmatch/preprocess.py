
from torch.nn import Module
from typing import Any, Callable, Tuple


class ReMixMatchUnlabeledPreProcess(Module):
	"""
		Compose transform_weak and transform_strong for unlabeled data.

		Note: (weak(data), (strong(data), strong(data), ...))
	"""
	def __init__(self, transform_weak: Callable, transform_strong: Callable, num_augms: int):
		super().__init__()
		self.transform_weak = transform_weak
		self.transform_strong = transform_strong
		self.num_augms = num_augms

	def forward(self, data: Any) -> Tuple[Any, ...]:
		return self.transform_weak(data), tuple(self.transform_strong(data) for _ in range(self.num_augms))
