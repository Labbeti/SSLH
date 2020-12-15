
import copy
import torch

from metric_utils.metrics import Metrics

from mlu.nn import EMA
from mlu.utils.printers import ColumnPrinter, PrinterABC

from ssl.models.checkpoint import CheckPointABC
from ssl.utils.recorder.recorder_abc import RecorderABC
from ssl.utils.types import IterableSized
from ssl.validater_abc import ValidaterABC

from torch.nn import Module
from typing import Callable, Dict, Optional


class ValidaterEMA(ValidaterABC):
	def __init__(
		self,
		model: Module,
		activation: Callable,
		loader: IterableSized,
		metrics: Dict[str, Metrics],
		recorder: RecorderABC,
		checkpoint: Optional[CheckPointABC] = None,
		checkpoint_metric: Optional[str] = None,
		display: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "val",
		decay: float = 0.99,
	):
		"""
			Main class for running a validation with the exponential moving average of the parameters of a model on a dataset.

			:param model: The model to test.
			:param activation: The activation function of the model.
			:param loader: The validation dataloader.
			:param metrics: The dict of Metrics to apply between prediction and true label.
			:param recorder: The recorder object for saving scores.
			:param checkpoint: The checkpoint object for saving the best model according a specific score. (default: None)
			:param checkpoint_metric: The name of the metric to use for saving the best model. (default: None)
			:param display: The printer object for printing validation results during the loop. (default: ColumnPrinter())
			:param device: The torch.device to use for validation. (default: torch.device("cuda"))
			:param name: The name of the validation. (default: "val")
			:param decay: The EMA parameter for the validation model. (default: 0.99)
		"""
		super().__init__()
		self.model = model
		self.activation = activation
		self.loader = loader
		self.metrics = metrics
		self.recorder = recorder
		self.display = display
		self.checkpoint = checkpoint
		self.checkpoint_metric = checkpoint_metric
		self.device = device
		self.name = name

		self.ema_model = copy.deepcopy(model)
		self.ema = EMA(self.ema_model, decay=decay)

	def _val_impl(self, epoch: int):
		self.model.eval()
		for metric in self.metrics.values():
			metric.reset()

		self.recorder.start_record(epoch)

		# TODO : update ema_model with ema, but when ?

		for i, (x, y) in enumerate(self.loader):
			x = x.to(self.device).float()
			y = y.to(self.device).float()

			logits = self.model(x)
			pred = self.activation(logits, dim=1)

			# Compute metrics
			for metric_name, metric in self.metrics.items():
				_mean = metric(pred, y)
				self.recorder.add_point("{:s}/{:s}".format(self.name, metric_name), metric.value.item())

			self.display.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch)

		if self.checkpoint is not None:
			current_means = self.recorder.get_current_means()
			if self.checkpoint_metric in current_means.keys():
				self.checkpoint.step(current_means[self.checkpoint_metric])

		self.recorder.end_record(epoch)
