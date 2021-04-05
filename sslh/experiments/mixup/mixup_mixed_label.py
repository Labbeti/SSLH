
import torch

from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Optional, Tuple

from mlu.metrics import MetricDict
from mlu.nn import CrossEntropyWithVectors
from sslh.experiments.mixup.mixup import MixUp


class MixUpMixedLabel(MixUp):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Module = Softmax(dim=-1),
		criterion: Module = CrossEntropyWithVectors(reduction="mean"),
		alpha: float = 0.4,
		metric_dict_train: Optional[MetricDict] = None,
		metric_dict_val: Optional[MetricDict] = None,
		metric_dict_test: Optional[MetricDict] = None,
		log_on_epoch: bool = True,
	):
		super().__init__(
			model=model,
			optimizer=optimizer,
			activation=activation,
			criterion=criterion,
			alpha=alpha,
			metric_dict_train=metric_dict_train,
			metric_dict_val=metric_dict_val,
			metric_dict_test=metric_dict_test,
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
			self.log_dict({"train/loss": loss}, **self.log_params)

			scores = self.metric_dict_train(self.activation(self.model(xs)), ys)
			self.log_dict(scores, **self.log_params)

		return loss
