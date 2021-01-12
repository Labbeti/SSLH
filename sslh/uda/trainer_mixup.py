
import torch

from mlu.metrics import Metric
from mlu.utils.printers import ColumnPrinter, PrinterABC
from mlu.utils.misc import get_lr

from sslh.augments.mixup import MixUp
from sslh.uda.loss import UDALoss
from sslh.uda.trainer import UDATrainer
from sslh.utils.recorder.base import RecorderABC
from sslh.utils.types import IterableSized

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class UDATrainerMixUp(UDATrainer):
	def __init__(
		self,
		model: Module,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics_s: Dict[str, Metric],
		metrics_u: Dict[str, Metric],
		recorder: RecorderABC,
		criterion: Callable = UDALoss(),
		printer: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "train",
		temperature: float = 0.5,
		lambda_s: float = 1.0,
		lambda_u: float = 1.0,
		threshold: float = 0.8,
		alpha: float = 0.75,
	):
		"""
			MixMatch trainer.

			:param model: The pytorch model to train.
			:param activation: The activation function of the model. (Inputs: (x: Tensor, dim: int)).
			:param optim: The optimizer used to update the model.
			:param criterion: The loss function.
			:param loader: The dataloader used to load ((batch_s, labels_s), (batch_u, batch_u_augm)).
			:param metrics_s: Metrics used during training for predictions on labeled data.
			:param metrics_u: Metrics used during training for predictions on unlabeled data.
			:param recorder: The recorder used to store metrics. (default: CrossEntropyWithVectors())
			:param printer: The object used to print values during training. (default: ColumnPrinter())
			:param device: The Pytorch device used for tensors. (default: torch.device("cuda"))
			:param name: The name of the training. (default: "train")
			:param temperature: The temperature used in sharpening function for Pseudo-labeling.
			:param lambda_s: The coefficient of labeled loss component.
			:param lambda_u: The coefficient of unlabeled loss component.
			:param threshold: The confidence threshold used for compute unsupervised loss or not.
			:param alpha: The MixUp alpha hyperparameter used for generating the mixup_lambda coefficient for mixing batchs and labels.
		"""
		super().__init__(
			model, activation, optim, loader, metrics_s, metrics_u, recorder, criterion, printer, device, name, temperature, lambda_s, lambda_u, threshold
		)
		self.mixup = MixUp(alpha, apply_max=True)

	def _train_impl(self, epoch: int):
		self.model.train()
		self.recorder.add_scalar("train/lr", get_lr(self.optim))

		for i, ((batch_s, labels_s), (batch_u, batch_u_augm)) in enumerate(self.loader):
			batch_s = batch_s.to(self.device).float()
			labels_s = labels_s.to(self.device).float()
			batch_u = batch_u.to(self.device).float()
			batch_u_augm = batch_u_augm.to(self.device).float()

			# Use guess u label with prediction of weak augmentation of u
			with torch.no_grad():
				labels_u, pred_u_augm = self.guess_label(batch_u_augm)
				mask = self.confidence_mask(pred_u_augm, self.threshold, dim=1)

				batch_s_mix, labels_s_mix = self.mixup(batch_s, batch_u, labels_s, labels_u)
				batch_u_mix, labels_u_mix = self.mixup(batch_u, batch_s, labels_u, labels_s)

			self.optim.zero_grad()

			# Compute predictions
			logits_s_mix = self.model(batch_s_mix)
			logits_u_mix = self.model(batch_u_mix)

			pred_s_mix = self.activation(logits_s_mix, dim=1)
			pred_u_mix = self.activation(logits_u_mix, dim=1)

			# Update model
			loss, loss_s, loss_u = self.criterion(
				pred_s_mix,
				pred_u_mix,
				mask,
				labels_s_mix,
				labels_u_mix,
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
					score = metric(pred_s_mix, labels_s_mix)
					self.recorder.add_scalar(metric_name, score)

				for metric_name, metric in self.metrics_u.items():
					score = metric(pred_u_mix, labels_u_mix)
					self.recorder.add_scalar(metric_name, score)

				self.printer.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch, self.name)
