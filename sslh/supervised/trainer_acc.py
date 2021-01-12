import torch

from mlu.metrics import Metric
from mlu.utils.printers import ColumnPrinter, PrinterABC
from mlu.utils.misc import get_lr

from sslh.supervised.loss import CrossEntropyWithVectors
from sslh.supervised.trainer import SupervisedTrainer
from sslh.utils.recorder.base import RecorderABC
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
		metrics: Dict[str, Metric],
		recorder: RecorderABC,
		criterion: Callable = CrossEntropyWithVectors(),
		printer: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "train",
		backward_frequency: int = 1,
	):
		"""
			Supervised trainer with accumulative gradients.

			:param model: The pytorch model to train.
			:param activation: The activation function of the model. (Inputs: (x: Tensor, dim: int)).
			:param optim: The optimizer used to update the model.
			:param criterion: The loss function.
			:param loader: The dataloader used to load (batch_s, labels_s).
			:param metrics: Metrics used during training.
			:param recorder: The recorder used to store metrics. (default: CrossEntropyWithVectors())
			:param printer: The object used to print values during training. (default: ColumnPrinter())
			:param device: The Pytorch device used for tensors. (default: torch.device("cuda"))
			:param name: The name of the training. (default: "train")
			:param backward_frequency: The frequency of the backward process.
		"""
		super().__init__(
			model, activation, optim, loader, metrics, recorder, criterion, printer, device,
		)
		self.backward_frequency = backward_frequency
		self.counter = 0

	def _train_impl(self, epoch: int):
		self.model.train()
		self.recorder.add_scalar("train/lr", get_lr(self.optim))

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
				self.recorder.add_scalar("train/loss", loss.item())

				for metric_name, metric in self.metrics.items():
					score = metric(pred, y)
					self.recorder.add_scalar(metric_name, score)

				self.printer.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch, self.name)

		if self.counter > 0:
			self.optim.step()
			self.optim.zero_grad()
			self.counter = 0
