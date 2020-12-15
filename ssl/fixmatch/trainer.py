import torch

from mlu.utils.printers import ColumnPrinter, PrinterABC
from mlu.utils.misc import get_lr

from ssl.fixmatch.loss import FixMatchLoss
from ssl.trainer_abc import TrainerABC
from ssl.utils.other_metrics import Metrics
from ssl.utils.recorder.recorder_abc import RecorderABC
from ssl.utils.types import IterableSized

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class FixMatchTrainer(TrainerABC):
	def __init__(
		self,
		model: Module,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics_s: Dict[str, Metrics],
		metrics_u: Dict[str, Metrics],
		recorder: RecorderABC,
		criterion: Callable = FixMatchLoss(),
		display: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		threshold: float = 0.5,
		lambda_s: float = 1.0,
		lambda_u: float = 1.0,
	):
		"""
			FixMatch trainer.

			:param model: The pytorch model to train.
			:param activation: The activation function of the model. (Inputs: (x: Tensor, dim: int)).
			:param optim: The optimizer used to update the model.
			:param loader: The dataloader used to load ((batch_s_weak, labels_s), (batch_u_weak, batch_u_strong))
			:param metrics_s: Metrics used during training on prediction labeled and labels.
			:param metrics_u: Metrics used during training on prediction unlabeled and labels.
			:param recorder: The recorder used to store metrics.
			:param criterion: The loss function.
			:param display: The object used to print values during training.
			:param device: The Pytorch device used for tensors.
			:param threshold: The confidence threshold used for compute unsupervised loss or not.
			:param lambda_s: The coefficient of labeled loss component.
			:param lambda_u: The coefficient of unlabeled loss component.
		"""
		super().__init__()
		self.model = model
		self.activation = activation
		self.optim = optim
		self.criterion = criterion
		self.loader = loader
		self.metrics_s = metrics_s
		self.metrics_u = metrics_u
		self.recorder = recorder
		self.display = display
		self.device = device
		self.threshold = threshold
		self.lambda_s = lambda_s
		self.lambda_u = lambda_u

	def _train_impl(self, epoch: int):
		self.model.train()
		for metrics in (self.metrics_s.values(), self.metrics_u.values()):
			for metric in metrics:
				metric.reset()

		self.recorder.start_record(epoch)

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
				self.recorder.add_point("train/loss", loss.item())
				self.recorder.add_point("train/loss_s", loss_s.item())
				self.recorder.add_point("train/loss_u", loss_u.item())
				self.recorder.add_point("train/labels_used", mask.mean().item())

				for metric_name, metric in self.metrics_s.items():
					_mean = metric(pred_s_augm_weak, labels_s)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				for metric_name, metric in self.metrics_u.items():
					_mean = metric(pred_u_augm_strong, labels_u)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				self.display.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch)

		self.recorder.add_point("train/lr", get_lr(self.optim))
		self.recorder.end_record(epoch)

	def guess_label(self, batch_u_augm_weak: Tensor) -> (Tensor, Tensor):
		logits_u_augm_weak = self.model(batch_u_augm_weak)
		pred_u_augm_weak = self.activation(logits_u_augm_weak, dim=1)

		nb_classes = pred_u_augm_weak.shape[1]
		labels_u = one_hot(pred_u_augm_weak.argmax(dim=1), nb_classes)
		return labels_u, pred_u_augm_weak

	def confidence_mask(self, pred_weak: Tensor, threshold: float, dim: int) -> Tensor:
		max_values, _ = pred_weak.max(dim=dim)
		return (max_values > threshold).float()
