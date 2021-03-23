
import torch

from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Callable, Optional, Tuple

from mlu.metrics import MetricDict
from mlu.nn import CrossEntropyWithVectors, OneHot
from sslh.transforms.augments.mixup import MixUpModule
from sslh.experiments.fixmatch.fixmatch import FixMatch


class FixMatchMixUp(FixMatch):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Callable = Softmax(dim=-1),
		criterion_s: Module = CrossEntropyWithVectors(reduction="none"),
		criterion_u: Module = CrossEntropyWithVectors(reduction="none"),
		metric_dict_train_s: Optional[MetricDict] = None,
		metric_dict_train_u_pseudo: Optional[MetricDict] = None,
		metric_dict_val: Optional[MetricDict] = None,
		metric_dict_test: Optional[MetricDict] = None,
		log_on_epoch: bool = True,
		target_transform: Callable = OneHot(num_classes=10),
		lambda_u: float = 1.0,
		threshold: float = 0.95,
		alpha: float = 0.75,
	):
		super().__init__(
			model=model,
			optimizer=optimizer,
			activation=activation,
			criterion_s=criterion_s,
			criterion_u=criterion_u,
			metric_dict_train_s=metric_dict_train_s,
			metric_dict_train_u_pseudo=metric_dict_train_u_pseudo,
			metric_dict_val=metric_dict_val,
			metric_dict_test=metric_dict_test,
			log_on_epoch=log_on_epoch,
			target_transform=target_transform,
			lambda_u=lambda_u,
			threshold=threshold,
		)

		self.alpha = alpha
		self.mixup = MixUpModule(alpha=alpha, apply_max=True)
		self.save_hyperparameters("alpha")

	def training_step(
		self,
		batch: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]],
		batch_idx: int,
	):
		(xs_weak, ys), (xu_weak, xu_strong) = batch

		# Compute pseudo-labels "yu" and mask
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
			self.log_dict(
				{"train/loss": loss, "train/loss_s": loss_s, "train/loss_u": loss_u, "train/mask": mask.mean()},
				**self.log_params
			)

			pred_xs_weak = self.activation(self.model(xs_weak))
			scores_s = self.metric_dict_train_s(pred_xs_weak, ys)
			self.log_dict(scores_s, **self.log_params)

			pred_xu_strong = self.activation(self.model(xu_strong))
			scores_u = self.metric_dict_train_u_pseudo(pred_xu_strong, yu)
			self.log_dict(scores_u, **self.log_params)

		return loss
