
import torch

from sslh.utils.torch import CrossEntropyWithVectors

from torch import Tensor
from torch.nn import Module


class ReMixMatchLoss(Module):
	def __init__(self):
		super().__init__()
		self.criterion_s = CrossEntropyWithVectors()
		self.criterion_u = CrossEntropyWithVectors()
		self.criterion_u1 = CrossEntropyWithVectors()
		self.criterion_r = CrossEntropyWithVectors()

	def forward(
		self,
		pred_s: Tensor,
		pred_u: Tensor,
		pred_u1: Tensor,
		pred_r: Tensor,
		labels_s: Tensor,
		labels_u: Tensor,
		labels_u1: Tensor,
		labels_r: Tensor,
		lambda_s: float,
		lambda_u: float,
		lambda_u1: float,
		lambda_r: float,
	) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
		"""
			Compute the ReMixMatch loss with 4 components.

			:param pred_s: Prediction of the supervised mixed batch "s_mix".
			:param pred_u: Prediction of the unsupervised mixed batch "u_mix".
			:param pred_u1: Prediction of the unsupervised batch "u1".
			:param pred_r: Prediction of the self-supervised component (rotation in original ReMixMatch) for batch "u1".
			:param labels_s: One-hot mixed labels for supervised mixed batch "s_mix".
			:param labels_u: One-hot mixed labels of the unsupervised mixed batch "u_mix".
			:param labels_u1: One-hot labels of the unsupervised batch "u1".
			:param labels_r: One-hot labels of the self-supervised component for batch "u1".
			:param lambda_s: Hyperparameter coefficient for mixed supervised loss.
			:param lambda_u: Hyperparameter coefficient for mixed unsupervised loss.
			:param lambda_u1: Hyperparameter coefficient for unsupervised loss.
			:param lambda_r: Hyperparameter coefficient for self-supervised loss.
		"""

		loss_s = self.criterion_s(pred_s, labels_s)
		loss_u = self.criterion_u(pred_u, labels_u)
		loss_u1 = self.criterion_u1(pred_u1, labels_u1)
		loss_r = self.criterion_r(pred_r, labels_r)

		loss = lambda_s * loss_s + lambda_u * loss_u + lambda_u1 * loss_u1 + lambda_r * loss_r

		return loss, loss_s, loss_u, loss_u1, loss_r
