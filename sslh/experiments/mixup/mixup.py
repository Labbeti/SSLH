
import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Callable, Optional, Tuple

from mlu.metrics import MetricDict
from mlu.nn import CrossEntropyWithVectors
from sslh.transforms.augments.mixup import MixUpModule


class MixUp(LightningModule):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Callable = Softmax(dim=-1),
		criterion: Module = CrossEntropyWithVectors(reduction="mean"),
		metric_dict_train: Optional[MetricDict] = None,
		metric_dict_val: Optional[MetricDict] = None,
		metric_dict_test: Optional[MetricDict] = None,
		log_on_epoch: bool = True,
		alpha: float = 0.4,
	):
		if metric_dict_train is None:
			metric_dict_train = MetricDict()
		if metric_dict_val is None:
			metric_dict_val = MetricDict()
		if metric_dict_test is None:
			metric_dict_test = MetricDict()

		super().__init__()
		self.model = model
		self.activation = activation
		self.optimizer = optimizer
		self.metric_dict_train = metric_dict_train
		self.metric_dict_val = metric_dict_val
		self.metric_dict_test = metric_dict_test
		self.criterion = criterion
		self.alpha = alpha

		self.mixup = MixUpModule(alpha=alpha, apply_max=False)

		self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)
		self.save_hyperparameters("alpha")

	def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
		xs, ys = batch

		with torch.no_grad():
			indexes = torch.randperm(xs.shape[0])
			xs_shuffle = xs[indexes]
			ys_shuffle = ys[indexes]

			xs_mix, ys_mix = self.mixup(xs, xs_shuffle, ys, ys_shuffle)
			lambda_ = self.mixup.get_last_lambda()

		pred_xs_mix = self.activation(self.model(xs_mix))
		loss = lambda_ * self.criterion(pred_xs_mix, ys) + (1.0 - lambda_) * self.criterion(pred_xs_mix, ys_shuffle)

		with torch.no_grad():
			self.log_dict({"train/loss": loss}, **self.log_params)

			scores = self.metric_dict_train(self.model(xs), ys)
			self.log_dict(scores, **self.log_params)

		return loss

	def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_xs = self.activation(self.model(xs))
		self.log_dict(self.metric_dict_val(pred_xs, ys), on_epoch=True, on_step=False)

	def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_xs = self.activation(self.model(xs))
		self.log_dict(self.metric_dict_test(pred_xs, ys), on_epoch=True, on_step=False)

	def forward(self, x: Tensor) -> Tensor:
		pred_x = self.activation(self.model(x))
		return pred_x

	def configure_optimizers(self) -> Optimizer:
		return self.optimizer
