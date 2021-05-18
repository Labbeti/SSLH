
import torch

from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Dict, Optional, Tuple

from mlu.nn import CrossEntropyWithVectors
from sslh.expt.mixup.mixup import MixUp


class MixUpMixLabel(MixUp):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Module = Softmax(dim=-1),
		criterion: Module = CrossEntropyWithVectors(),
		alpha: float = 0.4,
		train_metrics: Optional[Dict[str, Module]] = None,
		val_metrics: Optional[Dict[str, Module]] = None,
		log_on_epoch: bool = True,
	):
		"""
			MixUp with mix labels (MUM) LightningModule.

			:param model: The PyTorch Module to train.
				The forward() must return logits for classify the data.
			:param optimizer: The PyTorch optimizer to use.
			:param activation: The activation function of the model.
				(default: Softmax(dim=-1))
			:param criterion: The criterion used for compute loss.
				(default: CrossEntropyWithVectors())
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
			criterion=criterion,
			alpha=alpha,
			train_metrics=train_metrics,
			val_metrics=val_metrics,
			log_on_epoch=log_on_epoch,
		)

	def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
		xs, ys = batch

		with torch.no_grad():
			indexes = torch.randperm(xs.shape[0])
			xs_shuffle = xs[indexes]
			ys_shuffle = ys[indexes]

			xs_mix, ys_mix = self.mixup(xs, xs_shuffle, ys, ys_shuffle)

		pred_xs_mix = self.activation(self.model(xs_mix))
		loss = self.criterion(pred_xs_mix, ys_mix)

		with torch.no_grad():
			scores = {'train/loss': loss}
			scores = {k: v.cpu() for k, v in scores.items()}
			self.log_dict(scores, **self.log_params)

			scores = self.metric_dict_train(self.activation(self.model(xs)), ys)
			self.log_dict(scores, **self.log_params)

		return loss
