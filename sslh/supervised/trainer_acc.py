import torch

from metric_utils.metrics import Metrics

from sslh.supervised.loss import CrossEntropyWithVectors
from sslh.supervised.trainer import SupervisedTrainer
from mlu.utils.printers import ColumnPrinter
from mlu.utils.printers import PrinterABC
from sslh.utils.recorder.recorder_abc import RecorderABC
from sslh.utils.types import IterableSized

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class SupervisedTrainerAcc(SupervisedTrainer):
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
		backward_frequency: int = 1,
	):
		super().__init__(
			model, activation, optim, loader, metrics, recorder, criterion, display, device,
		)
		self.backward_frequency = backward_frequency
		self.counter = 0

	def _train_impl(self, epoch: int):
		self.model.train()
		for metric in self.metrics.values():
			metric.reset()

		self.recorder.start_record(epoch)

		for i, (x, y) in enumerate(self.loader):
			x = x.to(self.device).float()
			y = y.to(self.device).float()

			logits = self.model(x)
			pred = self.activation(logits, dim=1)

			loss = self.criterion(pred, y)
			loss.backward(retain_graph=True)

			self.counter += 1
			if self.counter >= self.backward_frequency:
				self.optim.step()
				self.optim.zero_grad()
				self.counter = 0

			# Compute metrics
			with torch.no_grad():
				self.recorder.add_point("train/loss", loss.item())

				for metric_name, metric in self.metrics.items():
					_mean = metric(pred, y)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				self.display.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch)

		self.recorder.end_record(epoch)

		if self.counter > 0:
			self.optim.step()
			self.optim.zero_grad()
			self.counter = 0
