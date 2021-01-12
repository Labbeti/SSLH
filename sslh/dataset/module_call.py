
from torch.nn import Module
from typing import Callable


class ModuleCall(Module, Callable):
	"""
		Wrapper for a Callable object with a forward() method.
	"""
	def __init__(self, callable_: Callable):
		super().__init__()
		self.callable_ = callable_

	def forward(self, *args, **kwargs):
		return self.callable_(*args, **kwargs)
