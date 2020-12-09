import torch

from abc import ABC
from sslh.caller import Caller


class ValidaterABC(Caller, ABC):
	"""
		Abstract class for validators.
		All subclasses must implements "_val_impl(self, epoch: int)" method.
		Note that all code in "_val_impl" is inside a "with torch.no_grad()" statement.
	"""
	def __init__(self):
		super().__init__()

	def _val_impl(self, epoch: int):
		raise NotImplementedError("Abstract method")

	def val(self, epoch: int):
		"""
			Start validation process.
		"""
		self._on_start()
		with torch.no_grad():
			self._val_impl(epoch)
		self._on_end()
