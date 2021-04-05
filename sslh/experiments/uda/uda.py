import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Optional, Tuple

from mlu.metrics import MetricDict
from mlu.nn import CrossEntropyWithVectors


class UDA(LightningModule):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Module = Softmax(dim=-1),
		criterion_s: Module = CrossEntropyWithVectors(reduction="none"),
		criterion_u: Module = CrossEntropyWithVectors(reduction="none"),
		lambda_u: float = 1.0,
		threshold: float = 0.8,
		temperature: float = 0.4,
		metric_dict_train_s: Optional[MetricDict] = None,
		metric_dict_train_u_pseudo: Optional[MetricDict] = None,
		metric_dict_val: Optional[MetricDict] = None,
		metric_dict_test: Optional[MetricDict] = None,
		log_on_epoch: bool = True,
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
		self.optimizer = optimizer
		self.activation = activation
		self.criterion_s = criterion_s
		self.criterion_u = criterion_u
		self.metric_dict_train_s = metric_dict_train_s
		self.metric_dict_train_u_pseudo = metric_dict_train_u_pseudo
		self.metric_dict_val = metric_dict_val
		self.metric_dict_test = metric_dict_test
		self.lambda_u = lambda_u
		self.threshold = threshold
		self.temperature = temperature

		self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)

		self.save_hyperparameters({
			"experiment": self.__class__.__name__,
			"model": model.__class__.__name__,
			"activation": activation.__class__.__name__,
			"optimizer": optimizer.__class__.__name__,
			"criterion_s": criterion_s.__class__.__name__,
			"criterion_u": criterion_u.__class__.__name__,
			"lambda_u": lambda_u,
			"threshold": threshold,
			"temperature": temperature,
		})

	def training_step(
		self,
		batch: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]],
		batch_idx: int,
	):
		(xs, ys), (xu, xu_strong) = batch

		# Compute pseudo-labels "yu" and mask
		yu, mask = self.guess_label_and_mask(xu)

		# Compute predictions on xs and xu
		pred_xs = self.activation(self.model(xs))
		pred_xu_strong = self.activation(self.model(xu_strong))

		# Criterion (loss_s of shape bsize_s, loss_u of shape bsize_u)
		loss_s = self.criterion_s(pred_xs, ys)
		loss_u = self.criterion_u(pred_xu_strong, yu)

		loss_s = torch.mean(loss_s)
		loss_u = torch.mean(loss_u * mask)

		loss = loss_s + self.lambda_u * loss_u
		import math
		if math.isnan(loss) or math.isinf(loss):
			breakpoint()

		with torch.no_grad():
			self.log_dict(
				{"train/loss": loss, "train/loss_s": loss_s, "train/loss_u": loss_u, "train/mask": mask.mean()},
				**self.log_params
			)

			scores_s = self.metric_dict_train_s(pred_xs, ys)
			self.log_dict(scores_s, **self.log_params)

			pred_xu = self.activation(self.model(xu))
			scores_u = self.metric_dict_train_u_pseudo(pred_xu, yu)
			self.log_dict(scores_u, **self.log_params)

		return loss

	def guess_label_and_mask(self, xu: Tensor) -> Tuple[Tensor, Tensor]:
		with torch.no_grad():
			logits_xu = self.model(xu)
			pred_xu = self.activation(logits_xu)
			probabilities_max, _ = pred_xu.max(dim=-1)
			mask = probabilities_max.ge(self.threshold).to(pred_xu.dtype)
			yu = torch.softmax(logits_xu / self.temperature, dim=-1)
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
		return self.activation(self.model(x))

	def configure_optimizers(self) -> Optimizer:
		return self.optimizer
