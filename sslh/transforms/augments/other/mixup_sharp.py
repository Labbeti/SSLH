
from torch import Tensor
from sslh.transforms.augments.mixup import MixUpModule


class MixUpSharp(MixUpModule):
	def __init__(self, alpha: float = 0.75, apply_max: bool = True, temperature: float = 0.3):
		super().__init__(alpha, apply_max)
		# Note: temperature in [0, 1], near 0 increase lambda a lot, near 1 increase a little
		self.temperature = temperature

	def forward(self, xa: Tensor, xb: Tensor, ya: Tensor, yb: Tensor) -> (Tensor, Tensor):
		if xa.shape != xb.shape or ya.shape != yb.shape:
			raise RuntimeError("Invalid shapes for MixUp : ({:s} != {:s} or {:s} != {:s})".format(
				xa.shape, xb.shape, ya.shape, yb.shape))

		self._lambda = self._sample()
		if self.apply_max:
			self._lambda = max(self._lambda, 1.0 - self._lambda)

		batch_mix = xa * self._lambda + xb * (1.0 - self._lambda)
		labels_mix = ya * self._lambda ** self.temperature + yb * (1.0 - self._lambda) ** self.temperature

		return batch_mix, labels_mix
