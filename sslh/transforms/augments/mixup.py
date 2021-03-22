
from torch import Tensor
from torch.distributions.beta import Beta
from torch.nn import Module


class MixUpModule(Module):
	"""
		Module MixUp that mix batch and labels with a parameter lambda sample from a beta distribution.

		Code overview :

		lambda ~ Beta(alpha, alpha) \n
		lambda = max(lambda, 1 - lambda) \n
		batch = batch_a * lambda + batch_b * (1 - lambda) \n
		label = label_a * lambda + label_b * (1 - lambda) \n

		Note:
			- if alpha -> 0 and apply_max == True, lambda sampled near 1,
			- if alpha -> 1 and apply_max == True, lambda sampled from an uniform distribution in [0.5, 1.0],
			- if alpha -> 0 and apply_max == False, lambda sampled near 1 or 0,
			- if alpha -> 1 and apply_max == False, lambda sampled from an uniform distribution in [0.0, 1.0],
	"""

	def __init__(self, alpha: float = 0.4, apply_max: bool = False):
		"""
			Build the MixUp Module.

			:param alpha: Controls the Beta distribution used to sampled the coefficient lambda. (default: 0.4)
			:param apply_max: If True, apply the "lambda = max(lambda, 1 - lambda)" after the sampling of lambda. (default: False)
				This operation is useful for having a mixed batch near to the first batch passed as input.
				It was set to True in MixMatch training but not in original MixUp training.
		"""
		super().__init__()
		self.alpha = alpha
		self.apply_max = apply_max

		self._beta = Beta(alpha, alpha)
		self._lambda = 0.0

	def forward(self, xa: Tensor, xb: Tensor, ya: Tensor, yb: Tensor) -> (Tensor, Tensor):
		"""
			Apply MixUp to batches and labels.
		"""
		if xa.shape != xb.shape or ya.shape != yb.shape:
			raise RuntimeError("Invalid shapes for MixUp : ({} != {} or {} != {})".format(
				xa.shape, xb.shape, ya.shape, yb.shape))

		# Sample from Beta distribution
		self._lambda = self._beta.sample().item() if self.alpha > 0.0 else 1.0

		if self.apply_max:
			self._lambda = max(self._lambda, 1.0 - self._lambda)

		batch_mix = xa * self._lambda + xb * (1.0 - self._lambda)
		labels_mix = ya * self._lambda + yb * (1.0 - self._lambda)

		return batch_mix, labels_mix

	def get_last_lambda(self) -> float:
		"""
			:returns: the last lambda sampled. If no data has been passed to forward(), returns 0.0.
		"""
		return self._lambda
