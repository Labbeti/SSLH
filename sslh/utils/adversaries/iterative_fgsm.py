from sslh.utils.adversaries.fgsm import FGSM

from torch import Tensor
from torch.nn import Module
from typing import Callable, Optional


class IterativeFGSM(Module):
	"""
		Iterative Gradient Sign
	"""

	def __init__(
		self,
		model: Module,
		activation: Callable,
		criterion: Callable,
		epsilon: float,
		nb_iterations: int,
		clip_min: Optional[float] = None,
		clip_max: Optional[float] = None,
	):
		super().__init__()
		self.fgsm = FGSM(
			model, activation, criterion, epsilon, clip_min, clip_max
		)
		self.nb_iterations = nb_iterations

	def forward(self, batch: Tensor, labels: Tensor) -> Tensor:
		perturbed_batch = batch

		for _ in range(self.nb_iterations):
			perturbed_batch = self.fgsm(perturbed_batch, labels)

		return perturbed_batch
