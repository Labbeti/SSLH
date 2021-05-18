
import torch

from sslh.transforms.augments.mixup import MixUpModule

from torch import Tensor
from torch.distributions.beta import Beta
from torch.distributions.uniform import Uniform
from torch.nn import Module


class MixUpRoll(Module):
	def __init__(self, dim_roll: int = -1, alpha: float = 0.4, apply_max: bool = False):
		super().__init__()
		self.dim_roll = dim_roll
		self.alpha = alpha
		self.apply_max = apply_max

		self._beta = Beta(alpha, alpha)
		self._lambda = 0.0
		self._uniform = Uniform(low=0, high=1)

	def forward(self, batch_a: Tensor, batch_b: Tensor, labels_a: Tensor, labels_b: Tensor) -> (Tensor, Tensor):
		if batch_a.shape != batch_b.shape:
			raise RuntimeError(f'Mismatch between batch shapes for MixUpRoll : ({batch_a.shape} != {batch_b.shape})')
		if labels_a.shape != labels_b.shape:
			raise RuntimeError(f'Mismatch between label shapes for MixUpRoll : ({labels_a.shape} != {labels_b.shape})')

		# Sample from Beta distribution
		self._lambda = self._beta.sample().item() if self.alpha > 0.0 else 1.0

		if self.apply_max:
			self._lambda = max(self._lambda, 1.0 - self._lambda)

		shift_a = torch.floor(batch_a.shape[self.dim_roll] * self._uniform.sample()).int().item()
		batch_a = batch_a.roll(shifts=shift_a, dims=self.dim_roll)

		shift_b = torch.floor(batch_a.shape[self.dim_roll] * self._uniform.sample()).int().item()
		batch_b = batch_b.roll(shifts=shift_b, dims=self.dim_roll)

		batch_mix = batch_a * self._lambda + batch_b * (1.0 - self._lambda)
		labels_mix = labels_a * self._lambda + labels_b * (1.0 - self._lambda)

		return batch_mix, labels_mix

	def get_last_lambda(self) -> float:
		"""
			Returns the last lambda sampled. If no data has been passed to forward(), returns 0.0.
		"""
		return self._lambda


class MixUpRollBatchShuffle(Module):
	"""
		Apply MixUp transform with the same batch in a different order. See MixUp module for details.
	"""
	def __init__(self, alpha: float = 0.4, apply_max: bool = False):
		super().__init__()
		self.mixup = MixUpModule(alpha, apply_max)

	def forward(self, batch: Tensor, labels: Tensor) -> (Tensor, Tensor):
		assert batch.shape[0] == labels.shape[0]
		batch_size = batch.shape[0]
		indexes = torch.randperm(batch_size)
		batch_shuffle = batch[indexes]
		labels_shuffle = labels[indexes]
		return self.mixup(batch, batch_shuffle, labels, labels_shuffle)

	def get_last_lambda(self) -> float:
		return self.mixup.get_last_lambda()
