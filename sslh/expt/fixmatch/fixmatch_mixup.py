
import torch

from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Dict, Optional, Tuple

from mlu.nn import CrossEntropyWithVectors, OneHot
from sslh.transforms.augments.mixup import MixUpModule
from sslh.expt.fixmatch.fixmatch import FixMatch


class FixMatchMixUp(FixMatch):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Module = Softmax(dim=-1),
		criterion_s: Module = CrossEntropyWithVectors(reduction='none'),
		criterion_u: Module = CrossEntropyWithVectors(reduction='none'),
		target_transform: Module = OneHot(n_classes=10),
		lambda_u: float = 1.0,
		threshold: float = 0.95,
		alpha: float = 0.75,
		train_metrics: Optional[Dict[str, Module]] = None,
		val_metrics: Optional[Dict[str, Module]] = None,
		log_on_epoch: bool = True,
	):
		"""
			FixMatch with MixUp (FMM) LightningModule.

			:param model: The PyTorch Module to train.
				The forward() must return logits for classify the data.
			:param optimizer: The PyTorch optimizer to use.
			:param activation: The activation function of the model.
				(default: Softmax(dim=-1))
			:param criterion_s: The loss component 'L_s' of RMM.
				(default: CrossEntropyWithVectors())
			:param criterion_u: The loss component 'L_u' of RMM.
				(default: CrossEntropyWithVectors())
			:param target_transform: The target transform for convert binarized labels to vector probabilities.
				(default: OneHot(n_classes=10))
			:param lambda_u: The coefficient of the 'L_u' component.
				(default: 1.0)
			:param threshold: The confidence threshold 'tau' used for the mask of the 'L_u' component.
				(default: 0.95)
			:param alpha: The mixup alpha parameter. A higher value means a stronger mix between labeled and unlabeled data.
				(default: 0.75)
			:param train_metrics: An optional dictionary of metrics modules for training.
				(default: None)
			:param val_metrics: An optional dictionary of metrics modules for validation.
				(default: None)
			:param log_on_epoch: If True, log only the epoch means of each train metric score.
				(default: True)
		"""
		super().__init__(
			model=model,
			optimizer=optimizer,
			activation=activation,
			criterion_s=criterion_s,
			criterion_u=criterion_u,
			target_transform=target_transform,
			lambda_u=lambda_u,
			threshold=threshold,
			train_metrics=train_metrics,
			val_metrics=val_metrics,
			log_on_epoch=log_on_epoch,
		)

		self.alpha = alpha
		self.mixup = MixUpModule(alpha=alpha, apply_max=True)
		self.save_hyperparameters({'alpha': alpha})

	def training_step(
		self,
		batch: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]],
		batch_idx: int,
	):
		(xs_weak, ys), (xu_weak, xu_strong) = batch

		# Compute pseudo-labels 'yu' and mask
		with torch.no_grad():
			yu, mask = self.guess_label_and_mask(xu_weak)

			# Apply mixup twice
			xs_mix, ys_mix = self.mixup(xs_weak, xu_strong, ys, yu)
			xu_mix, yu_mix = self.mixup(xu_strong, xs_weak, yu, ys)

		# Compute predictions on xs and xu
		pred_xs_mix = self.activation(self.model(xs_mix))
		pred_xu_mix = self.activation(self.model(xu_mix))

		# Criterion (loss_s of shape bsize_s, loss_u of shape bsize_u)
		loss_s = self.criterion_s(pred_xs_mix, ys)
		loss_u = self.criterion_u(pred_xu_mix, yu)

		loss_s = torch.mean(loss_s)
		loss_u = torch.mean(loss_u * mask)

		loss = loss_s + self.lambda_u * loss_u

		with torch.no_grad():
			scores = {'train/loss': loss, 'train/loss_s': loss_s, 'train/loss_u': loss_u, 'train/mask': mask.mean()}
			scores = {k: v.cpu() for k, v in scores.items()}
			self.log_dict(scores, **self.log_params)

			pred_xs_weak = self.activation(self.model(xs_weak))
			scores_s = self.metric_dict_train_s(pred_xs_weak, ys)
			self.log_dict(scores_s, **self.log_params)

			pred_xu_strong = self.activation(self.model(xu_strong))
			scores_u = self.metric_dict_train_u_pseudo(pred_xu_strong, yu)
			self.log_dict(scores_u, **self.log_params)

		return loss
