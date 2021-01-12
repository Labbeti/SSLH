
import torch

from mlu.metrics import Metric
from mlu.utils.printers import ColumnPrinter, PrinterABC
from mlu.utils.misc import get_lr

from sslh.augments.mixup_sharp import MixUpSharp
from sslh.supervised.loss import MixUpLoss
from sslh.trainer_abc import TrainerABC
from sslh.utils.recorder.base import RecorderABC
from sslh.utils.types import IterableSized

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class SupervisedTrainerMixUpMixLabelSharp(TrainerABC):
	def __init__(
		self,
		model: Module,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics: Dict[str, Metric],
		recorder: RecorderABC,
		criterion: Callable = MixUpLoss(),
		printer: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "train",
		alpha: float = 1.0,
		temperature: float = 0.3,
	):
		"""

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
			:param alpha: The MixUp alpha hyperparameter used for generating the mixup_lambda coefficient for mixing
				batches and labels. (default: 1.0)
			:param temperature: The temperature used to sharpen the labels.
		"""
		super().__init__()
		self.model = model
		self.activation = activation
		self.optim = optim
		self.loader = loader
		self.metrics = metrics
		self.recorder = recorder
		self.criterion = criterion
		self.printer = printer
		self.device = device
		self.name = name

		self.mixup = MixUpSharp(alpha, False, temperature)

	def _train_impl(self, epoch: int):
		self.model.train()
		self.recorder.add_scalar("train/lr", get_lr(self.optim))

		for i, (x, y) in enumerate(self.loader):
			x = x.to(self.device).float()
			y = y.to(self.device).float()

			with torch.no_grad():
				batch_size = x.shape[0]
				indexes = torch.randperm(batch_size)
				x_shuffle = x[indexes]
				y_shuffle = y[indexes]
				x_mix, y_mix = self.mixup(x, x_shuffle, y, y_shuffle)

			self.optim.zero_grad()

			logits_mix = self.model(x_mix)
			pred_mix = self.activation(logits_mix, dim=1)

			loss = self.criterion(pred_mix, y_mix)
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.recorder.add_scalar("train/loss", loss.item())

				for metric_name, metric in self.metrics.items():
					score = metric(pred_mix, y)
					self.recorder.add_scalar(metric_name, score)

				self.printer.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch, self.name)
