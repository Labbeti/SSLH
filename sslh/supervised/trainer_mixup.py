import torch

from metric_utils.metrics import Metrics

from sslh.supervised.loss import MixUpLoss
from sslh.supervised.mixup import MixUp
from sslh.trainer_abc import TrainerABC
from mlu.utils.printers import ColumnPrinter
from mlu.utils.printers import PrinterABC
from sslh.utils.recorder.recorder_abc import RecorderABC
from sslh.utils.types import IterableSized

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class SupervisedTrainerMixUp(TrainerABC):
	def __init__(
		self,
		model: Module,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics: Dict[str, Metrics],
		recorder: RecorderABC,
		criterion: Callable = MixUpLoss(),
		display: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		alpha: float = 1.0,
	):
		"""
			Supervised trainer.

			:param model: The pytorch model to train.
			:param activation: The activation function of the model. (Inputs: (x: Tensor, dim: int)).
			:param optim: The optimizer used to update the model.
			:param criterion: The loss function.
			:param loader: The dataloader used to load ((batch_s_weak, labels_s), (batch_u_weak, batch_u_strong))
			:param metrics: Metrics used during training.
			:param recorder: The recorder used to store metrics.
			:param display: The object used to print values during training.
			:param device: The Pytorch device used for tensors.
			:param alpha: The MixUp alpha hyperparameter used for generating the mixup_lambda coefficient for mixing batchs and labels.
		"""
		super().__init__()
		self.model = model
		self.activation = activation
		self.optim = optim
		self.loader = loader
		self.metrics = metrics
		self.recorder = recorder
		self.criterion = criterion
		self.display = display
		self.device = device

		self.mixup = MixUp(alpha, apply_max=False)

	def _train_impl(self, epoch: int):
		self.model.train()
		for metric in self.metrics.values():
			metric.reset()

		self.recorder.start_record(epoch)

		for i, (x, y) in enumerate(self.loader):
			x = x.to(self.device).float()
			y = y.to(self.device).float()

			with torch.no_grad():
				batch_size = x.shape[0]
				indexes = torch.randperm(batch_size)
				x_shuffle = x[indexes]
				y_shuffle = y[indexes]
				x_mix, _ = self.mixup(x, x_shuffle, y, y_shuffle)
				mixup_lambda = self.mixup.get_last_lambda()

			self.optim.zero_grad()

			logits_mix = self.model(x_mix)
			pred_mix = self.activation(logits_mix, dim=1)

			loss = self.criterion(pred_mix, y, y_shuffle, self.mixup.get_last_lambda())
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.recorder.add_point("train/loss", loss.item())
				self.recorder.add_point("train/mixup_lambda", mixup_lambda)

				for metric_name, metric in self.metrics.items():
					_mean = metric(pred_mix, y)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				self.display.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch)

		self.recorder.end_record(epoch)
