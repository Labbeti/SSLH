import torch

from metric_utils.metrics import Metrics

from sslh.supervised.loss import CrossEntropyWithVectors
from sslh.trainer_abc import TrainerABC
from sslh.utils.display import ColumnDisplay
from sslh.utils.display_abc import DisplayABC
from sslh.utils.recorder.recorder_abc import RecorderABC
from sslh.utils.types import IterableSized

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class SupervisedTrainer(TrainerABC):
	def __init__(
		self,
		model: Module,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics: Dict[str, Metrics],
		recorder: RecorderABC,
		criterion: Callable = CrossEntropyWithVectors(),
		display: DisplayABC = ColumnDisplay(),
		device: torch.device = torch.device("cuda"),
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

	def _train_impl(self, epoch: int):
		self.model.train()
		for metric in self.metrics.values():
			metric.reset()

		self.recorder.start_record(epoch)
		keys = list(self.metrics.keys()) + ["loss"]
		self.display.print_header("train", keys)

		iter_loader = iter(self.loader)

		for i, (x, y) in enumerate(iter_loader):
			x = x.to(self.device).float()
			y = y.to(self.device).float()

			self.optim.zero_grad()

			logits = self.model(x)
			pred = self.activation(logits, dim=1)

			loss = self.criterion(pred, y)
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.recorder.add_point("train/loss", loss.item())

				for metric_name, metric in self.metrics.items():
					_mean = metric(pred, y)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				self.display.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch)

		self.recorder.end_record(epoch)
