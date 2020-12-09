
import torch

from sslh.utils.display import ColumnDisplay
from sslh.utils.display_abc import DisplayABC
from sslh.utils.types import IterableSized
from sslh.validater_abc import ValidaterABC

from torch import IntTensor
from torch.nn import Module
from typing import Callable


class Validater(ValidaterABC):
	"""
		Main class for running a simple validation of a model on a dataset.
	"""

	def __init__(
		self,
		model: Module,
		activation: Callable,
		loader: IterableSized,
		nb_classes: int,
		display: DisplayABC = ColumnDisplay(),
		device: torch.device = torch.device("cuda"),
		name: str = "conf",
	):
		super().__init__()
		self.model = model
		self.activation = activation
		self.loader = loader
		self.display = display
		self.nb_classes = nb_classes
		self.device = device
		self.name = name

		self.matrix = IntTensor(torch.empty(0, dtype=torch.int))

	def _val_impl(self, epoch: int):
		self.model.eval()
		self.display.print_header(self.name, [], False)

		iter_loader = iter(self.loader)

		self.matrix = IntTensor(torch.zeros(self.nb_classes, self.nb_classes, dtype=torch.int))

		for i, (x, y) in enumerate(iter_loader):
			x = x.to(self.device).float()
			y = y.to(self.device).float()

			logits = self.model(x)
			pred = self.activation(logits, dim=1)

			pred = pred.argmax(dim=1)
			y = y.argmax(dim=1)

			self.matrix[pred, y] += 1

			self.display.print_current_values({}, i, len(self.loader), epoch)

	def get_matrix(self) -> IntTensor:
		return self.matrix
