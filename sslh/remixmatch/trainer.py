import torch

from mlu.metrics import Metric
from mlu.utils.printers import ColumnPrinter, PrinterABC
from mlu.utils.misc import get_lr

from sslh.mixmatch.sharpen import sharpen
from sslh.mixmatch.trainer import MixMatchTrainer
from sslh.remixmatch.average_pred import AveragePred
from sslh.remixmatch.loss import ReMixMatchLoss
from sslh.remixmatch.module_rot import ModuleRot
from sslh.utils.recorder.base import RecorderABC
from sslh.utils.torch import normalized
from sslh.utils.types import IterableSized

from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class ReMixMatchTrainer(MixMatchTrainer):
	def __init__(
		self,
		model: ModuleRot,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics_s_mix: Dict[str, Metric],
		metrics_u_mix: Dict[str, Metric],
		metrics_u1: Dict[str, Metric],
		metrics_r: Dict[str, Metric],
		recorder: RecorderABC,
		transform_self_supervised: Callable,
		criterion: Callable = ReMixMatchLoss(),
		printer: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "train",
		temperature: float = 0.5,
		alpha: float = 0.75,
		lambda_s: float = 1.0,
		lambda_u: float = 1.5,
		lambda_u1: float = 0.5,
		lambda_r: float = 0.5,
		history: int = 128,
	):
		"""
			ReMixMatch trainer.

			:param model: The pytorch model to train.
			:param activation: The activation function of the model. (Inputs: (x: Tensor, dim: int)).
			:param optim: The optimizer used to update the model.
			:param criterion: The loss function.
			:param loader: The dataloader used to load ((batch_s_weak, labels_s), (batch_u_weak, batch_u_strong))
			:param metrics_s_mix: Metrics used during training on mixed prediction labeled and labels.
			:param metrics_u_mix: Metrics used during training on mixed prediction unlabeled and labels.
			:param recorder: The recorder used to store metrics.
			:param printer: The object used to print values during training.
			:param device: The Pytorch device used for tensors.
			:param name: The name of the training.
			:param temperature: The temperature used in sharpening function for Pseudo-labeling.
			:param alpha: The alpha hyperparameter for MixUp.
			:param lambda_s: The coefficient of labeled loss component.
			:param lambda_u: The coefficient of unlabeled loss component.
			:param lambda_u1: The coefficient of unlabeled strong loss component.
			:param lambda_r: The coefficient of rotation loss component.
			:param history: The history size for compute the classes distribution.
		"""
		super().__init__(
			model=model,
			activation=activation,
			optim=optim,
			loader=loader,
			metrics_s_mix=metrics_s_mix,
			metrics_u_mix=metrics_u_mix,
			recorder=recorder,
			criterion=criterion,
			printer=printer,
			device=device,
			name=name,
			temperature=temperature,
			alpha=alpha,
			lambda_s=lambda_s,
			lambda_u=lambda_u,
		)
		self.metrics_u1 = metrics_u1
		self.metrics_r = metrics_r
		self.transform_self_supervised = transform_self_supervised
		self.lambda_u1 = lambda_u1
		self.lambda_r = lambda_r

		self.average_pred_s = AveragePred(history)
		self.average_pred_u = AveragePred(history)

		if not hasattr(self.model, "forward_rot"):
			raise RuntimeError("Model must have a rotation layer for predict a random rotation applied.")

	def _train_impl(self, epoch: int):
		self.model.train()
		self.recorder.add_scalar("train/lr", get_lr(self.optim))

		for i, ((batch_s_augm_strong, labels_s), (batch_u_augm_weak, batch_u_augm_strong_multiple)) in enumerate(self.loader):
			batch_s_augm_strong = batch_s_augm_strong.to(self.device).float()
			labels_s = labels_s.to(self.device).float()
			batch_u_augm_weak = batch_u_augm_weak.to(self.device).float()
			batch_u_augm_strong_multiple = torch.stack(batch_u_augm_strong_multiple).to(self.device).float()

			with torch.no_grad():
				self.average_pred_s.add_pred(labels_s)
				self.average_pred_u.add_pred(self.activation(self.model(batch_u_augm_weak), dim=1))

				labels_u = self.guess_label(batch_u_augm_weak, self.temperature)

				batch_s_mix, batch_u_mix, labels_s_mix, labels_u_mix = self.remixmatch(
					batch_s_augm_strong, batch_u_augm_weak, batch_u_augm_strong_multiple, labels_s, labels_u
				)

				# Get strongly augmented batch "batch_u1"
				batch_u1 = batch_u_augm_strong_multiple[0, :].clone()
				labels_u1 = labels_u.clone()

				batch_r, labels_r = self.transform_self_supervised(batch_u1)

			self.optim.zero_grad()

			logits_s_mix = self.model(batch_s_mix)
			logits_u_mix = self.model(batch_u_mix)
			logits_u1 = self.model(batch_u1)
			logits_r = self.model.forward_rot(batch_r)

			pred_s_mix = self.activation(logits_s_mix, dim=1)
			pred_u_mix = self.activation(logits_u_mix, dim=1)
			pred_u1 = self.activation(logits_u1, dim=1)
			pred_r = self.activation(logits_r, dim=1)

			loss, loss_s, loss_u, loss_u1, loss_r = self.criterion(
				pred_s_mix,
				pred_u_mix,
				pred_u1,
				pred_r,
				labels_s_mix,
				labels_u_mix,
				labels_u1,
				labels_r,
				self.lambda_s,
				self.lambda_u,
				self.lambda_u1,
				self.lambda_r,
			)
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.recorder.add_scalar("train/loss", loss.item())
				self.recorder.add_scalar("train/loss_s", loss_s.item())
				self.recorder.add_scalar("train/loss_u", loss_u.item())
				self.recorder.add_scalar("train/loss_u1", loss_u1.item())
				self.recorder.add_scalar("train/loss_r", loss_r.item())

				for metric_name, metric in self.metrics_s_mix.items():
					score = metric(pred_s_mix, labels_s_mix)
					self.recorder.add_scalar(metric_name, score)

				for metric_name, metric in self.metrics_u_mix.items():
					score = metric(pred_u_mix, labels_u_mix)
					self.recorder.add_scalar(metric_name, score)

				for metric_name, metric in self.metrics_u1.items():
					score = metric(pred_u1, labels_u1)
					self.recorder.add_scalar(metric_name, score)

				for metric_name, metric in self.metrics_r.items():
					score = metric(pred_r, labels_r)
					self.recorder.add_scalar(metric_name, score)

				self.printer.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch, self.name)

	def guess_label(self, batch_u_augm_weak: Tensor, temperature: float) -> Tensor:
		logits_u_augm_weak = self.model(batch_u_augm_weak)
		labels_u = self.activation(logits_u_augm_weak, dim=1)

		labels_u = labels_u * self.average_pred_s.get_mean() / self.average_pred_u.get_mean()
		labels_u = normalized(labels_u, dim=1)

		labels_u = sharpen(labels_u, temperature, dim=1)
		return labels_u

	def remixmatch(
		self,
		batch_s_augm_strong: Tensor,
		batch_u_augm_weak: Tensor,
		batch_u_augm_strong_multiple: Tensor,
		labels_s: Tensor,
		labels_u: Tensor,
	):
		# Concatenate strongly and weakly augmented data from batch_u
		batch_u_augm_weak = batch_u_augm_weak.reshape(1, *batch_u_augm_weak.shape)
		batch_u_multiple = torch.cat((batch_u_augm_strong_multiple, batch_u_augm_weak), dim=0)

		return self.mixmatch(batch_s_augm_strong, batch_u_multiple, labels_s, labels_u)
