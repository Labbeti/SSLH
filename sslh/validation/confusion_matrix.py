
import torch

from mlu.utils.printers import ColumnPrinter
from mlu.utils.printers import PrinterABC
from sslh.utils.types import IterableSized
from sslh.validater_abc import ValidaterABC

from torch import IntTensor
from torch.nn import Module
from typing import Callable


class ConfusionMatrix(ValidaterABC):
	"""
		Main class for running a simple validation of a model on a dataset.
	"""

	def __init__(
		self,
		model: Module,
		activation: Callable,
		loader: IterableSized,
		nb_classes: int,
		printer: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "conf",
	):
		super().__init__()
		self.model = model
		self.activation = activation
		self.loader = loader
		self.printer = printer
		self.nb_classes = nb_classes
		self.device = device
		self.name = name

		self.matrix = IntTensor(torch.empty(0, dtype=torch.int))

	def _val_impl(self, epoch: int):
		self.model.eval()

		self.matrix = IntTensor(torch.zeros(self.nb_classes, self.nb_classes, dtype=torch.int))

		for i, (x, y) in enumerate(self.loader):
			x = x.to(self.device).float()
			y = y.to(self.device).float()

			logits = self.model(x)
			pred = self.activation(logits, dim=1)

			pred = pred.argmax(dim=1)
			y = y.argmax(dim=1)

			self.matrix[pred, y] += 1

			self.printer.print_current_values({}, i, len(self.loader), epoch, self.name)

	def get_matrix(self) -> IntTensor:
		return self.matrix
