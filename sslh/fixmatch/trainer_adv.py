
import math
import torch

from advertorch.attacks import GradientSignAttack

from mlu.nn import CrossEntropyWithVectors
from mlu.utils.misc import get_lr
from mlu.utils.printers import ColumnPrinter, PrinterABC

from sslh.fixmatch.loss import FixMatchLoss
from sslh.fixmatch.trainer import FixMatchTrainer
from sslh.utils.other_metrics import Metrics, CategoricalAccuracyOnehot
from sslh.utils.recorder.recorder_abc import RecorderABC
from sslh.utils.types import IterableSized

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class FixMatchTrainerAdv(FixMatchTrainer):
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
		criterion_adv: Callable = CrossEntropyWithVectors(),
		epsilon_adv_weak: float = 1e-2,
		epsilon_adv_strong: float = 1e-1,
	):
		super().__init__(
			model, activation, optim, loader, metrics_s, metrics_u, recorder, criterion, display, device, threshold, lambda_s, lambda_u,
		)

		self.adv_acc_metric = CategoricalAccuracyOnehot(dim=1)

		criterion_adv_with_logits = lambda input_, target: criterion_adv(activation(input_, dim=1), target)
		self.adv_generator_weak = GradientSignAttack(
			predict=model,
			loss_fn=criterion_adv_with_logits,
			eps=epsilon_adv_weak,
			clip_min=-math.inf,
			clip_max=math.inf,
		)
		self.adv_generator_strong = GradientSignAttack(
			predict=model,
			loss_fn=criterion_adv_with_logits,
			eps=epsilon_adv_strong,
			clip_min=-math.inf,
			clip_max=math.inf,
		)

	def _train_impl(self, epoch: int):
		self.model.train()
		for metrics in (self.metrics_s.values(), self.metrics_u.values()):
			for metric in metrics:
				metric.reset()

		self.recorder.start_record(epoch)

		for i, ((batch_s, labels_s), batch_u) in enumerate(self.loader):
			batch_s = batch_s.to(self.device).float()
			labels_s = labels_s.to(self.device).float()
			batch_u = batch_u.to(self.device).float()

			# Use guess u label with prediction of weak augmentation of u
			with torch.no_grad():
				labels_u, pred_u = self.guess_label(batch_u)
				mask = self.confidence_mask(pred_u, self.threshold, dim=1)

			batch_s_adv_weak = self.adv_generator_weak(batch_s, labels_s)
			batch_u_adv_strong = self.adv_generator_strong(batch_u, labels_u)

			self.optim.zero_grad()

			# Compute predictions
			logits_s_adv_weak = self.model(batch_s_adv_weak)
			logits_u_adv_strong = self.model(batch_u_adv_strong)

			pred_s_adv_weak = self.activation(logits_s_adv_weak, dim=1)
			pred_u_adv_strong = self.activation(logits_u_adv_strong, dim=1)

			# Update model
			loss, loss_s, loss_u = self.criterion(
				pred_s_adv_weak,
				pred_u_adv_strong,
				mask,
				labels_s,
				labels_u,
				lambda_s=self.lambda_s,
				lambda_u=self.lambda_u,
			)

			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.recorder.add_point("train/loss", loss.item())
				self.recorder.add_point("train/loss_s", loss_s.item())
				self.recorder.add_point("train/loss_u", loss_u.item())

				pred_u_adv_strong = self.activation(self.model(batch_u_adv_strong), dim=1)
				self.adv_acc_metric(pred_u_adv_strong, labels_u)
				self.recorder.add_point("train/adv_acc", self.adv_acc_metric.value.item())
				self.recorder.add_point("train/labels_used", mask.mean().item())

				for metric_name, metric in self.metrics_s.items():
					_mean = metric(pred_s_adv_weak, labels_s)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				for metric_name, metric in self.metrics_u.items():
					_mean = metric(pred_u_adv_strong, labels_u)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				self.display.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch)

		self.recorder.add_point("train/lr", get_lr(self.optim))
		self.recorder.end_record(epoch)
