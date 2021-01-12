
from abc import ABC
from sslh.caller import Caller


class TrainerABC(Caller, ABC):
	"""
		Abstract class for trainers.

		Abstract methods:
			- _train_impl(self, epoch: int):
	"""
	def __init__(self):
		super().__init__()

	def _train_impl(self, epoch: int):
		raise NotImplementedError("Abstract method")

	def train(self, epoch: int):
		"""
			Start training process.

			:param epoch: The current epoch number.
		"""
		self._on_start()
		self._train_impl(epoch)
		self._on_end()
