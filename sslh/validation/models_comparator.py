
import torch

from mlu.metrics import Metric
from mlu.utils.printers import ColumnPrinter, PrinterABC

from sslh.utils.recorder import Recorder
from sslh.utils.types import IterableSized
from sslh.validater_abc import ValidaterABC

from torch import IntTensor
from torch.nn import Module
from typing import Callable, Dict, List


class ModelsComparator(ValidaterABC):
	"""
		Main class for running a simple validation of a model on a dataset.
	"""

	def __init__(
		self,
		model: List[Module],
		activation: Callable,
		loader: IterableSized,
		metrics: Dict[str, Metric],
		printer: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "comp",
	):
		super().__init__()
		self.models = model
		self.activation = activation
		self.loader = loader
		self.metrics = metrics
		self.printer = printer
		self.device = device
		self.name = name

		self.recorder = Recorder()
		self.matrix = IntTensor(torch.empty(0, dtype=torch.int))

		self.add_callback_on_end(self.recorder)

	def _val_impl(self, epoch: int):
		for model in self.models:
			model.eval()

		self.matrix = IntTensor(torch.zeros(2 ** len(self.models), dtype=torch.int))

		for i, (x, y) in enumerate(self.loader):
			x = x.to(self.device).float()
			y = y.to(self.device).float()

			batch_size = y.shape[0]
			y = y.argmax(dim=1)

			results = torch.zeros(len(self.models), batch_size, dtype=torch.int)
			for j, model in enumerate(self.models):
				pred = self.activation(model(x), dim=1)
				pred = pred.argmax(dim=1)
				results[j] = y.eq(pred).int()

				for metric_name, metric in self.metrics.items():
					score = metric(pred, y)
					self.recorder.add_scalar(f"{metric_name}_model{j}", score)

			for j in range(batch_size):
				corrects = results[:, j].tolist()
				idx = sum([correct * (2 ** k) for k, correct in enumerate(corrects)])
				self.matrix[idx] += 1

			self.printer.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch, self.name)

	def get_matrix(self) -> IntTensor:
		return self.matrix
