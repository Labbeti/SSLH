
from mlu.nn import CrossEntropyWithVectors

from torch import Tensor
from torch.nn import Module
from typing import Callable


class MixUpLoss(Module):
	def __init__(self, criterion: Callable = CrossEntropyWithVectors()):
		super().__init__()
		self.criterion = criterion

	def forward(self, pred: Tensor, labels_a: Tensor, labels_b: Tensor, lambda_: float) -> Tensor:
		"""
			:param pred: Output of the model for the mixed batch.
			:param labels_a: True labels without shuffle.
			:param labels_b: True labels with shuffle.
			:param lambda_: Coefficient used during the mix.
		"""
		return lambda_ * self.criterion(pred, labels_a) + (1.0 - lambda_) * self.criterion(pred, labels_b)


class MixUpLossSmooth(Module):
	def __init__(self, criterion: Callable = CrossEntropyWithVectors()):
		super().__init__()
		self.criterion = criterion

	def forward(self, pred: Tensor, labels_a: Tensor, labels_b: Tensor, lambda_: float) -> Tensor:
		"""
			:param pred: Output of the model for the mixed batch.
			:param labels_a: True labels without shuffle.
			:param labels_b: True labels with shuffle.
			:param lambda_: Coefficient used during the mix.
		"""
		return lambda_ * self.criterion(pred, labels_a * lambda_) + (1.0 - lambda_) * self.criterion(pred, labels_b * (1.0 - lambda_))
