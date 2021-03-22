
from torch.nn import Module
from typing import Any, Callable, Tuple


class FixMatchUnlabeledPreProcess(Module):
	"""
		Compose transform_weak and transform_strong for unlabeled data.

		Note: (weak(data), strong(data))
	"""
	def __init__(self, transform_weak: Callable, transform_strong: Callable):
		super().__init__()
		self.transform_weak = transform_weak
		self.transform_strong = transform_strong

	def forward(self, data: Any) -> Tuple[Any, Any]:
		return self.transform_weak(data), self.transform_strong(data)
