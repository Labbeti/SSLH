
import math
import torch

from advertorch.attacks import GradientSignAttack

from mlu.metrics import Metric, CategoricalAccuracy
from mlu.utils.printers import ColumnPrinter, PrinterABC
from mlu.utils.misc import get_lr

from sslh.supervised.loss import CrossEntropyWithVectors
from sslh.supervised.trainer import SupervisedTrainer
from sslh.utils.recorder.base import RecorderABC
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
		metrics: Dict[str, Metric],
		recorder: RecorderABC,
		criterion: Callable = CrossEntropyWithVectors(),
		printer: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "train",
		criterion_adv: Callable = CrossEntropyWithVectors(),
		epsilon_adv: float = 1e-2,
	):
		"""
			Supervised trainer with adversarial examples.

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
			:param criterion_adv: The criterion to use for generate adversaries. (default: CrossEntropyWithVectors())
			:param epsilon_adv: The epsilon to use for generate adversaries. (default: 1e-2)
		"""
		super().__init__(
			model, activation, optim, loader, metrics, recorder, criterion, printer, device, name
		)
		self.metrics_adv = {"train/acc_adv": CategoricalAccuracy(dim=1)}

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
		self.recorder.add_scalar("train/lr", get_lr(self.optim))

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
				self.recorder.add_scalar("train/loss", loss.item())

				for metric_name, metric in self.metrics_adv.items():
					score = metric(pred_adv, y)
					self.recorder.add_scalar(metric_name, score)

				for metric_name, metric in self.metrics.items():
					score = metric(pred_adv, y)
					self.recorder.add_scalar(metric_name, score)

				self.printer.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch, self.name)
