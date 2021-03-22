
import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Callable, Optional, Tuple

from mlu.metrics import MetricDict
from mlu.nn import CrossEntropyWithVectors


class Supervised(LightningModule):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Callable = Softmax(dim=-1),
		criterion: Module = CrossEntropyWithVectors(reduction="none"),
		metric_dict_train: Optional[MetricDict] = None,
		metric_dict_val: Optional[MetricDict] = None,
		metric_dict_test: Optional[MetricDict] = None,
		log_on_epoch: bool = True,
	):
		if metric_dict_train is None:
			metric_dict_train = MetricDict()
		if metric_dict_val is None:
			metric_dict_val = MetricDict()
		if metric_dict_test is None:
			metric_dict_test = MetricDict()

		super().__init__()
		self.model = model
		self.optimizer = optimizer
		self.activation = activation
		self.criterion = criterion
		self.metric_dict_train = metric_dict_train
		self.metric_dict_val = metric_dict_val
		self.metric_dict_test = metric_dict_test

		self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)

	def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
		xs, ys = batch

		pred_xs = self.activation(self.model(xs))
		loss = self.criterion(pred_xs, ys)

		with torch.no_grad():
			self.log_dict({"train/loss": loss}, **self.log_params)

			scores = self.metric_dict_train(pred_xs, ys)
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
