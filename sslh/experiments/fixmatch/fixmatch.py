
import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Callable, Optional, Tuple

from mlu.metrics import MetricDict
from mlu.nn import CrossEntropyWithVectors, OneHot


class FixMatch(LightningModule):
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
	):
		if metric_dict_train_s is None:
			metric_dict_train_s = MetricDict()
		if metric_dict_train_u_pseudo is None:
			metric_dict_train_u_pseudo = MetricDict()
		if metric_dict_val is None:
			metric_dict_val = MetricDict()
		if metric_dict_test is None:
			metric_dict_test = MetricDict()

		super().__init__()
		self.model = model
		self.activation = activation
		self.optimizer = optimizer
		self.target_transform = target_transform
		self.metric_dict_train_s = metric_dict_train_s
		self.metric_dict_train_u_pseudo = metric_dict_train_u_pseudo
		self.metric_dict_val = metric_dict_val
		self.metric_dict_test = metric_dict_test
		self.criterion_s = criterion_s
		self.criterion_u = criterion_u
		self.threshold = threshold
		self.lambda_u = lambda_u

		self.log_params = dict(on_epoch=True, on_step=not log_on_epoch)
		self.save_hyperparameters("lambda_u", "threshold")

	def training_step(
		self,
		batch: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]],
		batch_idx: int,
	):
		(xs_weak, ys), (xu_weak, xu_strong) = batch

		# Compute pseudo-labels "yu" and mask
		yu, mask = self.guess_label_and_mask(xu_weak)

		# Compute predictions on xs and xu
		pred_xs_weak = self.activation(self.model(xs_weak))
		pred_xu_strong = self.activation(self.model(xu_strong))

		# Criterion (loss_s of shape bsize_s, loss_u of shape bsize_u)
		loss_s = self.criterion_s(pred_xs_weak, ys)
		loss_u = self.criterion_u(pred_xu_strong, yu)

		loss_s = torch.mean(loss_s)
		loss_u = torch.mean(loss_u * mask)

		loss = loss_s + self.lambda_u * loss_u

		with torch.no_grad():
			self.log_dict({"train/loss": loss, "train/loss_s": loss_s, "train/loss_u": loss_u}, **self.log_params)

			scores_s = self.metric_dict_train_s(pred_xs_weak, ys)
			self.log_dict(scores_s, **self.log_params)

			scores_u = self.metric_dict_train_u_pseudo(pred_xu_strong, yu)
			self.log_dict(scores_u, **self.log_params)

		return loss

	def guess_label_and_mask(self, xu_weak: Tensor) -> Tuple[Tensor, Tensor]:
		with torch.no_grad():
			pred_xu_weak = self.activation(self.model(xu_weak))
			probabilities_max, indices_max = pred_xu_weak.max(dim=-1)
			mask = probabilities_max.ge(self.threshold).to(pred_xu_weak.dtype)
			yu = self.target_transform(indices_max)
			return yu, mask

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
