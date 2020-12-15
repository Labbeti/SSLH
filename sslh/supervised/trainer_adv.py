
import math
import torch

from advertorch.attacks import GradientSignAttack
from metric_utils.metrics import Metrics

from sslh.supervised.loss import CrossEntropyWithVectors
from sslh.supervised.trainer import SupervisedTrainer
from mlu.utils.printers import ColumnPrinter
from mlu.utils.printers import PrinterABC
from sslh.utils.other_metrics import CategoricalAccuracyOnehot
from sslh.utils.recorder.recorder_abc import RecorderABC
from sslh.utils.types import IterableSized

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class SupervisedTrainerAdv(SupervisedTrainer):
	def __init__(
		self,
		model: Module,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics: Dict[str, Metrics],
		recorder: RecorderABC,
		criterion: Callable = CrossEntropyWithVectors(),
		display: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		criterion_adv: Callable = CrossEntropyWithVectors(),
		epsilon_adv: float = 1e-2,
	):
		super().__init__(
			model, activation, optim, loader, metrics, recorder, criterion, display, device
		)
		self.adv_acc_metric = CategoricalAccuracyOnehot(dim=1)

		criterion_adv_with_logits = lambda input_, target: criterion_adv(activation(input_, dim=1), target)
		self.adv_generator = GradientSignAttack(
			predict=model,
			loss_fn=criterion_adv_with_logits,
			eps=epsilon_adv,
			clip_min=-math.inf,
			clip_max=math.inf,
		)

	def _train_impl(self, epoch: int):
		self.model.train()
		for metric in self.metrics.values():
			metric.reset()

		self.recorder.start_record(epoch)

		for i, (x, y) in enumerate(self.loader):
			x = x.to(self.device).float()
			y = y.to(self.device).float()

			x_adv = self.adv_generator(x, y)

			self.optim.zero_grad()

			logits_adv = self.model(x_adv)
			pred_adv = self.activation(logits_adv, dim=1)

			loss = self.criterion(pred_adv, y)
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.recorder.add_point("train/loss", loss.item())

				self.adv_acc_metric(pred_adv, y)
				self.recorder.add_point("train/adv_acc", self.adv_acc_metric.value.item())

				for metric_name, metric in self.metrics.items():
					_mean = metric(pred_adv, y)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				self.display.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch)

		self.recorder.end_record(epoch)
