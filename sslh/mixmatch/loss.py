
from sslh.utils.torch import CrossEntropyWithVectors

from torch import Tensor
from torch.nn import Module, MSELoss
from typing import Callable


class MixMatchLoss(Module):
	def __init__(
		self,
		criterion_s: Callable = CrossEntropyWithVectors(),
		criterion_u: Callable = MSELoss(reduction="mean")
	):
		super().__init__()
		self.criterion_s = criterion_s
		self.criterion_u = criterion_u

	def forward(
		self,
		pred_s: Tensor,
		pred_u: Tensor,
		labels_s: Tensor,
		labels_u: Tensor,
		lambda_s: float,
		lambda_u: float,
	) -> (Tensor, Tensor, Tensor):
		"""
			Compute MixMatch loss.

			Generic :
				loss = lambda_s * mean(criterion_s(pred_s, labels_s)) + lambda_u * mean(criterion_u(pred_u, labels_u))

			:param pred_s: Output of the model for labeled batch s of shape (batch_size, nb_classes).
			:param pred_u: Output of the model for unlabeled batch u of shape (batch_size, nb_classes).
			:param labels_s: True label of labeled batch s of shape (batch_size, nb_classes).
			:param labels_u: Guessed label of unlabeled batch u of shape (batch_size, nb_classes).
			:param lambda_s: Coefficient used to multiply the supervised loss component.
			:param lambda_u: Coefficient used to multiply the unsupervised loss component.
		"""
		loss_s = self.criterion_s(pred_s, labels_s)
		loss_u = self.criterion_u(pred_u, labels_u)

		loss = lambda_s * loss_s + lambda_u * loss_u

		return loss, loss_s, loss_u


class MixMatchLossNoLabelMix(Module):
	def __init__(self, criterion_s: Callable = CrossEntropyWithVectors(), criterion_u: Callable = CrossEntropyWithVectors()):
		super().__init__()
		self.criterion_s = criterion_s
		self.criterion_u = criterion_u

	def forward(
		self,
		pred_s: Tensor,
		pred_u: Tensor,
		labels_s: Tensor,
		labels_u: Tensor,
		labels_s_shuffle: Tensor,
		labels_u_shuffle: Tensor,
		lambda_s: float,
		lambda_u: float,
		mixup_lambda_s: float,
		mixup_lambda_u: float,
	) -> (Tensor, Tensor, Tensor):
		loss_s = mixup_lambda_s * self.criterion_s(pred_s, labels_s) + (1.0 - mixup_lambda_s) * self.criterion_s(pred_s, labels_s_shuffle)
		loss_u = mixup_lambda_u * self.criterion_u(pred_u, labels_u) + (1.0 - mixup_lambda_u) * self.criterion_u(pred_u, labels_u_shuffle)

		loss = lambda_s * loss_s + lambda_u * loss_u

		return loss, loss_s, loss_u
