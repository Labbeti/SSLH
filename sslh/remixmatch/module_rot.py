
from torch import Tensor
from typing import Protocol


class ModuleRot(Protocol):
	def __call__(self, x: Tensor) -> Tensor:
		raise NotImplementedError("Abstract method")

	def forward_rot(self, x: Tensor) -> Tensor:
		raise NotImplementedError("Abstract method")
