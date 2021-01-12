
from sslh.augments.mixup import MixUp
from torch import Tensor


class MixUpSharp(MixUp):
	def __init__(self, alpha: float = 0.75, apply_max: bool = True, temperature: float = 0.3):
		super().__init__(alpha, apply_max)
		# Note: temperature in [0, 1], near 0 increase lambda a lot, near 1 increase a little
		self.temperature = temperature

	def forward(self, batch_a: Tensor, batch_b: Tensor, labels_a: Tensor, labels_b: Tensor) -> (Tensor, Tensor):
		if batch_a.shape != batch_b.shape or labels_a.shape != labels_b.shape:
			raise RuntimeError("Invalid shapes for MixUp : ({:s} != {:s} or {:s} != {:s})".format(
				batch_a.shape, batch_b.shape, labels_a.shape, labels_b.shape))

		self._lambda = self._sample()
		if self.apply_max:
			self._lambda = max(self._lambda, 1.0 - self._lambda)

		batch_mix = batch_a * self._lambda + batch_b * (1.0 - self._lambda)
		labels_mix = labels_a * self._lambda ** self.temperature + labels_b * (1.0 - self._lambda) ** self.temperature

		return batch_mix, labels_mix
