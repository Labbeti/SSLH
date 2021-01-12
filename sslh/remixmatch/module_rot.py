
from torch import Tensor
from torch.nn import Module
from typing import Protocol, Union


class ModuleRotProtocol(Protocol):
	def __call__(self, x: Tensor) -> Tensor:
		raise NotImplementedError("Abstract method")

	def forward(self, x: Tensor) -> Tensor:
		raise NotImplementedError("Abstract method")

	def forward_rot(self, x: Tensor) -> Tensor:
		raise NotImplementedError("Abstract method")


ModuleRot = Union[Module, ModuleRotProtocol]
