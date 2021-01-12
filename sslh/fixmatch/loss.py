
from mlu.nn import CrossEntropyWithVectors, get_reduction_from_name

from torch import Tensor
from torch.nn import Module
from typing import Callable


class FixMatchLoss(Module):
	"""
		FixMatch loss module.

		Loss formula : loss = CE(pred_s, label_s) + lambda_u * mask * CE(pred_u, label_u)

		The mask used is 1 if the confidence prediction on weakly augmented data is above a specific threshold.
	"""

	def __init__(
		self,
		criterion_s: Callable = CrossEntropyWithVectors(reduction="none"),
		criterion_u: Callable = CrossEntropyWithVectors(reduction="none"),
		reduction: str = "mean",
	):
		"""
			:param criterion_s: The criterion used for labeled loss component.
			:param criterion_u: The criterion used for unlabeled loss component. No reduction must be applied.
			:param reduction: The main reduction to use. Can be 'none', 'mean' or 'sum'.
		"""
		super().__init__()
		self.criterion_s = criterion_s
		self.criterion_u = criterion_u
		self.reduce_fn = get_reduction_from_name(reduction)

	def forward(
		self,
		pred_s_augm_weak: Tensor,
		pred_u_augm_strong: Tensor,
		mask: Tensor,
		labels_s: Tensor,
		labels_u: Tensor,
		lambda_s: float = 1.0,
		lambda_u: float = 1.0,
	) -> (Tensor, Tensor, Tensor):
		"""
			Compute FixMatch loss.

			Generic :
				loss = lambda_s * mean(criterion_s(pred_s, labels_s)) + lambda_u * mean(criterion_u(pred_u, labels_u) * mask)

			:param pred_s_augm_weak: Output of the model for labeled batch s of shape (batch_size, nb_classes).
			:param pred_u_augm_strong: Output of the model for unlabeled batch u of shape (batch_size, nb_classes).
			:param mask: Binary confidence mask used to avoid using low-confidence labels as targets of shape (batch_size).
			:param labels_s: True label of labeled batch s of shape (batch_size, nb_classes).
			:param labels_u: Guessed label of unlabeled batch u of shape (batch_size, nb_classes).
			:param lambda_s: Coefficient used to multiply the supervised loss component.
			:param lambda_u: Coefficient used to multiply the unsupervised loss component.
		"""
		loss_s = self.criterion_s(pred_s_augm_weak, labels_s)

		loss_u = self.criterion_u(pred_u_augm_strong, labels_u)
		loss_u = loss_u * mask

		loss_s = self.reduce_fn(loss_s)
		loss_u = self.reduce_fn(loss_u)

		loss = lambda_s * loss_s + lambda_u * loss_u

		return loss, loss_s, loss_u


class FixMatchLossSoftReduceU(Module):
	"""
		FixMatch loss module with loss_u mean reduction using the number of pseudo labels used instead of constant bsize_u.

		Loss formula : loss = CE(pred_s, label_s) + lambda_u * mask * CE(pred_u, label_u)

		The mask used is 1 if the confidence prediction on weakly augmented data is above a specific threshold.
	"""

	def __init__(
		self,
		criterion_s: Callable = CrossEntropyWithVectors(reduction="none"),
		criterion_u: Callable = CrossEntropyWithVectors(reduction="none"),
		reduction: str = "mean",
	):
		"""
			:param criterion_s: The criterion used for labeled loss component.
			:param criterion_u: The criterion used for unlabeled loss component. No reduction must be applied.
			:param reduction: The main reduction to use. Can be 'none', 'mean' or 'sum'.
		"""
		super().__init__()
		self.criterion_s = criterion_s
		self.criterion_u = criterion_u
		self.reduce_fn = get_reduction_from_name(reduction)

	def forward(
		self,
		pred_s_augm_weak: Tensor,
		pred_u_augm_strong: Tensor,
		mask: Tensor,
		labels_s: Tensor,
		labels_u: Tensor,
		lambda_s: float = 1.0,
		lambda_u: float = 1.0,
	) -> (Tensor, Tensor, Tensor):
		"""
			Compute FixMatch loss.

			Generic :
				loss = lambda_s * mean(criterion_s(pred_s, labels_s)) + lambda_u * mean(criterion_u(pred_u, labels_u) * mask)

			:param pred_s_augm_weak: Output of the model for labeled batch s of shape (batch_size, nb_classes).
			:param pred_u_augm_strong: Output of the model for unlabeled batch u of shape (batch_size, nb_classes).
			:param mask: Binary confidence mask used to avoid using low-confidence labels as targets of shape (batch_size).
			:param labels_s: True label of labeled batch s of shape (batch_size, nb_classes).
			:param labels_u: Guessed label of unlabeled batch u of shape (batch_size, nb_classes).
			:param lambda_s: Coefficient used to multiply the supervised loss component.
			:param lambda_u: Coefficient used to multiply the unsupervised loss component.
		"""
		loss_s = self.criterion_s(pred_s_augm_weak, labels_s)

		loss_u = self.criterion_u(pred_u_augm_strong, labels_u)
		loss_u *= mask

		loss_s = self.reduce_fn(loss_s)
		loss_u = loss_u.sum()
		mask_sum = mask.sum()
		if mask_sum != 0.0:
			loss_u = loss_u / mask_sum

		loss = lambda_s * loss_s + lambda_u * loss_u

		return loss, loss_s, loss_u
