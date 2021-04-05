
import math
import torch

from advertorch.attacks import GradientSignAttack
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss, Softmax, LogSoftmax
from torch.optim import Optimizer
from typing import Optional, Tuple

from mlu.metrics import MetricDict
from mlu.nn import CrossEntropyWithVectors


class DeepCoTraining(LightningModule):
	def __init__(
		self,
		model_f: Module,
		model_g: Module,
		optimizer: Optimizer,
		activation: Module = Softmax(dim=-1),
		log_activation: Module = LogSoftmax(dim=-1),
		criterion_s: Module = CrossEntropyWithVectors(log_input=True),
		epsilon: float = 0.02,
		lambda_cot: float = 1.0,
		lambda_diff: float = 0.5,
		metric_dict_train_f: Optional[MetricDict] = None,
		metric_dict_train_g: Optional[MetricDict] = None,
		metric_dict_val_f: Optional[MetricDict] = None,
		metric_dict_val_g: Optional[MetricDict] = None,
		metric_dict_test_f: Optional[MetricDict] = None,
		metric_dict_test_g: Optional[MetricDict] = None,
		log_on_epoch: bool = True,
	):
		if metric_dict_train_f is None:
			metric_dict_train_f = MetricDict()

		if metric_dict_train_g is None:
			metric_dict_train_g = MetricDict()

		if metric_dict_val_f is None:
			metric_dict_val_f = MetricDict()

		if metric_dict_val_g is None:
			metric_dict_val_g = MetricDict()

		if metric_dict_test_f is None:
			metric_dict_test_f = MetricDict()

		if metric_dict_test_g is None:
			metric_dict_test_g = MetricDict()

		super().__init__()
		self.model_f = model_f
		self.model_g = model_g
		self.optimizer = optimizer
		self.activation = activation
		self.log_activation = log_activation
		self.criterion_s = criterion_s
		self.epsilon = epsilon
		self.lambda_cot = lambda_cot
		self.lambda_diff = lambda_diff
		self.metric_dict_train_f = metric_dict_train_f
		self.metric_dict_train_g = metric_dict_train_g
		self.metric_dict_val_f = metric_dict_val_f
		self.metric_dict_val_g = metric_dict_val_g
		self.metric_dict_test_f = metric_dict_test_f
		self.metric_dict_test_g = metric_dict_test_g
		self.log_on_epoch = log_on_epoch

		self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)

		gsa_params = dict(
			loss_fn=CrossEntropyLoss(reduction="sum"),
			eps=epsilon,
			clip_min=-math.inf,
			clip_max=math.inf,
			targeted=False,
		)
		self.adv_generator_f = GradientSignAttack(model_f, **gsa_params)
		self.adv_generator_g = GradientSignAttack(model_g, **gsa_params)

		self.save_hyperparameters({
			"experiment": self.__class__.__name__,
			"model": model_f.__class__.__name__,
			"optimizer": optimizer.__class__.__name__,
			"activation": activation.__class__.__name__,
			"log_activation": log_activation.__class__.__name__,
			"criterion_s": criterion_s.__class__.__name__,
			"epsilon": epsilon,
			"lambda_cot": lambda_cot,
			"lambda_diff": lambda_diff,
		})

	def training_step(
		self,
		batch: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor],
		batch_idx: int,
	):
		(xs1, ys1), (xs2, ys2), xu = batch

		logits_f_xs1 = self.model_f(xs1)
		logits_g_xs2 = self.model_g(xs2)
		logits_f_xu = self.model_f(xu)
		logits_g_xu = self.model_g(xu)

		with torch.no_grad():
			pred_f_xu = logits_f_xu.argmax(dim=-1)
			pred_g_xu = logits_g_xu.argmax(dim=-1)

			ys1_indices = ys1.argmax(dim=-1)
			ys2_indices = ys2.argmax(dim=-1)

		adv_f_xs1 = self.adv_generator_f.perturb(xs1, ys1_indices)
		adv_f_xu = self.adv_generator_f.perturb(xu, pred_f_xu)

		adv_g_xs2 = self.adv_generator_g.perturb(xs2, ys2_indices)
		adv_g_xu = self.adv_generator_g.perturb(xu, pred_g_xu)

		# Note: logits of model "f" for the adversarial example of model "g" of batch "xs2"
		logits_f_adv_g_xs2 = self.model_f(adv_g_xs2)
		logits_g_adv_f_xs1 = self.model_g(adv_f_xs1)

		logits_f_adv_g_xu = self.model_f(adv_g_xu)
		logits_g_adv_f_xu = self.model_g(adv_f_xu)

		# Compute losses
		loss_sup = self.loss_sup(logits_f_xs1, logits_g_xs2, ys1, ys2)
		loss_cot = self.loss_cot(logits_f_xu, logits_g_xu)
		loss_diff = self.loss_diff(
			logits_f_xs1, logits_g_xs2, logits_f_adv_g_xs2, logits_g_adv_f_xs1,
			logits_f_xu, logits_g_xu, logits_f_adv_g_xu, logits_g_adv_f_xu,
		)

		loss = loss_sup + self.lambda_cot * loss_cot + self.lambda_diff * loss_diff

		with torch.no_grad():
			# Compute metrics
			self.log_dict({"loss": loss, "loss_sup": loss_sup, "loss_cot": loss_cot, "loss_diff": loss_diff})

			scores_f = self.metric_dict_train_f(self.activation(logits_f_xs1), ys1)
			self.log_dict(scores_f, **self.log_params)

			scores_g = self.metric_dict_train_g(self.activation(logits_g_xs2), ys2)
			self.log_dict(scores_g, **self.log_params)

		return loss

	def loss_sup(self, logits_f_xs1: Tensor, logits_g_xs2: Tensor, ys1: Tensor, ys2: Tensor) -> Tensor:
		log_pred_f_xs1 = self.log_activation(logits_f_xs1)
		log_pred_g_xs2 = self.log_activation(logits_g_xs2)
		return self.criterion_s(log_pred_f_xs1, ys1) + self.criterion_s(log_pred_g_xs2, ys2)

	def loss_cot(self, logits_f_xu: Tensor, logits_g_xu: Tensor) -> Tensor:
		pred_f_xu = self.activation(logits_f_xu)
		pred_g_xu = self.activation(logits_g_xu)

		mean_pred = 0.5 * (pred_f_xu + pred_g_xu)
		mean_pred = torch.clamp(mean_pred, min=1e-8)

		loss_mean_pred = mean_pred * mean_pred.log()
		loss_mean_pred = -loss_mean_pred.sum()

		loss_f = self.activation(logits_f_xu) * self.log_activation(logits_f_xu)
		loss_f = -loss_f.sum()

		loss_g = self.activation(logits_g_xu) * self.log_activation(logits_g_xu)
		loss_g = -loss_g.sum()

		bsize_u = logits_f_xu.shape[0]
		loss_cot = loss_mean_pred - 0.5 * (loss_f + loss_g) / bsize_u

		return loss_cot

	def loss_diff(
		self,
		logits_f_xs1: Tensor,
		logits_g_xs2: Tensor,
		logits_f_adv_g_xs2: Tensor,
		logits_g_adv_f_xs1: Tensor,
		logits_f_xu: Tensor,
		logits_g_xu: Tensor,
		logits_f_adv_g_xu: Tensor,
		logits_g_adv_f_xu: Tensor,
	) -> Tensor:

		loss_g_xs2 = self.activation(logits_g_xs2) * self.log_activation(logits_f_adv_g_xs2)
		loss_g_xs2 = loss_g_xs2.sum()

		loss_f_xs1 = self.activation(logits_f_xs1) * self.log_activation(logits_g_adv_f_xs1)
		loss_f_xs1 = loss_f_xs1.sum()

		loss_g_xu = self.activation(logits_g_xu) * self.log_activation(logits_f_adv_g_xu)
		loss_g_xu = loss_g_xu.sum()

		loss_f_xu = self.activation(logits_f_xu) * self.log_activation(logits_g_adv_f_xu)
		loss_f_xu = loss_f_xu.sum()

		total_bsize = logits_f_xs1.shape[0] + logits_f_xu.shape[0]
		loss_diff = -(loss_g_xs2 + loss_f_xs1 + loss_g_xu + loss_f_xu) / total_bsize

		return loss_diff

	def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_f_xs = self.activation(self.model_f(xs))
		pred_g_xs = self.activation(self.model_g(xs))

		self.log_dict(self.metric_dict_val_f(pred_f_xs, ys))
		self.log_dict(self.metric_dict_val_g(pred_g_xs, ys))

	def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_f_xs = self.activation(self.model_f(xs))
		pred_g_xs = self.activation(self.model_g(xs))

		self.log_dict(self.metric_dict_test_f(pred_f_xs, ys))
		self.log_dict(self.metric_dict_test_g(pred_g_xs, ys))

	def forward(self, x: Tensor, model_used: str = "model_f") -> Tensor:
		"""
			TODO: Default use model f, maybe use g ?
		"""

		if model_used in ["f", "model_f"]:
			pred_x = self.activation(self.model_f(x))

		elif model_used in ["g", "model_g"]:
			pred_x = self.activation(self.model_g(x))

		elif model_used in ["mean"]:
			pred_f_x = self.activation(self.model_f(x))
			pred_g_x = self.activation(self.model_g(x))
			pred_x = (pred_f_x + pred_g_x) / 2.0

		elif model_used in ["most_confident"]:
			pred_f_x = self.activation(self.model_f(x))
			pred_g_x = self.activation(self.model_g(x))
			if pred_f_x.max() > pred_g_x.max():
				pred_x = pred_f_x
			else:
				pred_x = pred_g_x

		else:
			raise RuntimeError(
				f"Invalid model used '{model_used}'. "
				f"Must be one of {('f', 'model_f', 'g', 'model_g', 'mean', 'most_confident')}."
			)
		
		return pred_x

	def configure_optimizers(self) -> Optimizer:
		return self.optimizer

	def get_model_f(self) -> Module:
		return self.model_f

	def get_model_g(self) -> Module:
		return self.model_g
