
import copy
import torch

from mlu.metrics import Metric
from mlu.nn import EMA
from mlu.utils.misc import get_lr
from mlu.utils.printers import ColumnPrinter, PrinterABC

from sslh.fixmatch.loss import FixMatchLoss
from sslh.fixmatch.trainer import FixMatchTrainer
from sslh.utils.recorder.base import RecorderABC
from sslh.utils.types import IterableSized

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class FixMatchTrainerTeacher(FixMatchTrainer):
	def __init__(
		self,
		model: Module,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics_s: Dict[str, Metric],
		metrics_u: Dict[str, Metric],
		recorder: RecorderABC,
		criterion: Callable = FixMatchLoss(),
		printer: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "train",
		threshold: float = 0.95,
		lambda_s: float = 1.0,
		lambda_u: float = 1.0,
		decay: float = 0.999,
	):
		"""
			FixMatch trainer with Teacher.

			:param model: The pytorch model to train.
			:param activation: The activation function of the model. (Inputs: (x: Tensor, dim: int)).
			:param optim: The optimizer used to update the model.
			:param loader: The dataloader used to load ((batch_s_weak, labels_s), (batch_u_weak, batch_u_strong))
			:param metrics_s: Metrics used during training on prediction labeled and labels.
			:param metrics_u: Metrics used during training on prediction unlabeled and labels.
			:param recorder: The recorder used to store metrics.
			:param criterion: The loss function.
			:param printer: The object used to print values during training.
			:param device: The Pytorch device used for tensors.
			:param name: The name of the training.
			:param threshold: The confidence threshold used for compute unsupervised loss or not.
			:param lambda_s: The coefficient of labeled loss component.
			:param lambda_u: The coefficient of unlabeled loss component.
			:param decay: The decay used by exponential moving average.
		"""
		super().__init__(
			model,
			activation,
			optim,
			loader,
			metrics_s,
			metrics_u,
			recorder,
			criterion,
			printer,
			device,
			name,
			threshold,
			lambda_s,
			lambda_u,
		)
		self.teacher = copy.deepcopy(model)
		self.ema = EMA(self.teacher, decay=decay)

	def _train_impl(self, epoch: int):
		self.model.train()
		self.recorder.add_scalar("train/lr", get_lr(self.optim))

		for i, ((batch_s_augm_weak, labels_s), (batch_u_augm_weak, batch_u_augm_strong)) in enumerate(self.loader):
			batch_s_augm_weak = batch_s_augm_weak.to(self.device).float()
			labels_s = labels_s.to(self.device).float()
			batch_u_augm_weak = batch_u_augm_weak.to(self.device).float()
			batch_u_augm_strong = batch_u_augm_strong.to(self.device).float()

			# Use guess u label with prediction of weak augmentation of u
			with torch.no_grad():
				labels_u, pred_u_augm_weak = self.guess_label(batch_u_augm_weak)
				mask = self.confidence_mask(pred_u_augm_weak, self.threshold, dim=1)

			self.optim.zero_grad()

			# Compute predictions
			logits_s_augm_weak = self.model(batch_s_augm_weak)
			logits_u_augm_strong = self.model(batch_u_augm_strong)

			pred_s_augm_weak = self.activation(logits_s_augm_weak, dim=1)
			pred_u_augm_strong = self.activation(logits_u_augm_strong, dim=1)

			# Update model
			loss, loss_s, loss_u = self.criterion(
				pred_s_augm_weak,
				pred_u_augm_strong,
				mask,
				labels_s,
				labels_u,
				lambda_s=self.lambda_s,
				lambda_u=self.lambda_u
			)

			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.recorder.add_scalar("train/loss", loss.item())
				self.recorder.add_scalar("train/loss_s", loss_s.item())
				self.recorder.add_scalar("train/loss_u", loss_u.item())
				self.recorder.add_scalar("train/labels_used", mask.mean().item())

				for metric_name, metric in self.metrics_s.items():
					score = metric(pred_s_augm_weak, labels_s)
					self.recorder.add_scalar(metric_name, score)

				for metric_name, metric in self.metrics_u.items():
					score = metric(pred_u_augm_strong, labels_u)
					self.recorder.add_scalar(metric_name, score)

				self.printer.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch, self.name)
				self.ema.update(self.model)

	def guess_label(self, batch_u_augm_weak: Tensor) -> (Tensor, Tensor):
		logits_u_augm_weak = self.teacher(batch_u_augm_weak)
		pred_u_augm_weak = self.activation(logits_u_augm_weak, dim=1)

		nb_classes = pred_u_augm_weak.shape[1]
		labels_u = one_hot(pred_u_augm_weak.argmax(dim=1), nb_classes)
		return labels_u, pred_u_augm_weak
