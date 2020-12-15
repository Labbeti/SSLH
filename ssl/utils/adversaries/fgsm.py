
from torch import Tensor
from torch.nn import Module
from typing import Callable, Optional


class FGSM(Module):
	"""
		Fast Gradient Sign Attack
	"""

	def __init__(
		self,
		model: Module,
		activation: Callable,
		criterion: Callable,
		epsilon: float,
		clip_min: Optional[float] = None,
		clip_max: Optional[float] = None,
	):
		super().__init__()
		self.model = model
		self.activation = activation
		self.criterion = criterion
		self.epsilon = epsilon
		self.clip_min = clip_min
		self.clip_max = clip_max

	def forward(self, batch: Tensor, labels: Tensor) -> Tensor:
		batch = batch.detach().clone()
		batch.requires_grad = True

		logits = self.model(batch)
		pred = self.activation(logits, dim=1)

		loss = self.criterion(pred, labels)
		self.model.zero_grad()
		loss.backward()

		grad_sign = batch.grad.data.sign()
		perturbed_batch = batch + self.epsilon * grad_sign

		if self.clip_min is not None:
			perturbed_batch = perturbed_batch.clamp(min=self.clip_min)

		if self.clip_max is not None:
			perturbed_batch = perturbed_batch.clamp(max=self.clip_max)

		return perturbed_batch.detach()
