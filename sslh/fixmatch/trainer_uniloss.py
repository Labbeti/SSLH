import torch

from mlu.metrics import Metric
from mlu.nn import CrossEntropyWithVectors
from mlu.utils.printers import ColumnPrinter, PrinterABC
from mlu.utils.misc import get_lr

from sslh.fixmatch.loss import FixMatchLoss
from sslh.fixmatch.trainer import FixMatchTrainer
from sslh.utils.misc import interpolation
from sslh.utils.recorder.base import RecorderABC
from sslh.utils.types import IterableSized

from torch.distributions.categorical import Categorical
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict, List


class FixMatchTrainerUniLoss(FixMatchTrainer):
	def __init__(
		self,
		model: Module,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics_s: Dict[str, Metric],
		metrics_u: Dict[str, Metric],
		recorder: RecorderABC,
		nb_epochs: int,
		start_probs: List[float],
		target_probs: List[float],
		criterion: Callable = FixMatchLoss(),
		printer: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "train",
		threshold: float = 0.95,
		lambda_s: float = 1.0,
		lambda_u: float = 1.0,
	):
		"""
			FixMatch trainer with a random choice between compute labeled loss component and unlabeled loss component.

			:param model: The pytorch model to train.
			:param activation: The activation function of the model. (Inputs: (x: Tensor, dim: int)).
			:param optim: The optimizer used to update the model.
			:param loader: The dataloader used to load ((batch_s_weak, labels_s), (batch_u_weak, batch_u_strong))
			:param metrics_s: Metrics used during training on prediction labeled and labels.
			:param metrics_u: Metrics used during training on prediction unlabeled and labels.
			:param recorder: The recorder used to store metrics.
			:param criterion: The loss function.
			:param printer: The object used to print values during training.
			:param device: The Pytorch device used for tensors.
			:param name: The name of the training.
			:param threshold: The confidence threshold used for compute unsupervised loss or not.
			:param lambda_s: The coefficient of labeled loss component.
			:param lambda_u: The coefficient of unlabeled loss component.
		"""
		super().__init__(
			model,
			activation,
			optim,
			loader,
			metrics_s,
			metrics_u,
			recorder,
			criterion,
			printer,
			device,
			name,
			threshold,
			lambda_s,
			lambda_u,
		)
		self.nb_epochs = nb_epochs

		self.criterion_s = CrossEntropyWithVectors()
		self.criterion_u = CrossEntropyWithVectors(reduction="none")
		self.start_probs = torch.as_tensor(start_probs)
		self.target_probs = torch.as_tensor(target_probs)
		self.categorical = Categorical(probs=self.start_probs)
		self.probs = self.start_probs
		self.reduce_fn = torch.mean

	def _train_impl(self, epoch: int):
		self.model.train()
		self.recorder.add_scalar("train/lr", get_lr(self.optim))

		self.probs = torch.zeros(len(self.start_probs))
		for i, target in enumerate(self.target_probs):
			self.probs[i] = interpolation(self.start_probs[i], self.target_probs[i], epoch / self.nb_epochs)
		self.categorical = Categorical(probs=self.probs)
		self.recorder.add_scalar("train/prob_s", self.probs[0].item())
		self.recorder.add_scalar("train/prob_u", self.probs[1].item())

		for i, ((batch_s_augm_weak, labels_s), (batch_u_augm_weak, batch_u_augm_strong)) in enumerate(self.loader):
			loss_chosen = self.choose_loss()

			if loss_chosen == "s":
				batch_s_augm_weak = batch_s_augm_weak.to(self.device).float()
				labels_s = labels_s.to(self.device).float()

				self.optim.zero_grad()
				logits_s_augm_weak = self.model(batch_s_augm_weak)
				pred_s_augm_weak = self.activation(logits_s_augm_weak, dim=1)

				loss_s = self.criterion_s(pred_s_augm_weak, labels_s) * self.lambda_s
				loss_s.backward()
				self.optim.step()

				with torch.no_grad():
					self.recorder.add_scalar("train/loss_s", loss_s.item())

					for metric_name, metric in self.metrics_s.items():
						score = metric(pred_s_augm_weak, labels_s)
						self.recorder.add_scalar(metric_name, score)

			else:
				batch_u_augm_weak = batch_u_augm_weak.to(self.device).float()
				batch_u_augm_strong = batch_u_augm_strong.to(self.device).float()

				# Use guess u label with prediction of weak augmentation of u
				with torch.no_grad():
					labels_u, pred_u_augm_weak = self.guess_label(batch_u_augm_weak)
					mask = self.confidence_mask(pred_u_augm_weak, self.threshold, dim=1)

				self.optim.zero_grad()

				# Compute predictions
				logits_u_augm_strong = self.model(batch_u_augm_strong)
				pred_u_augm_strong = self.activation(logits_u_augm_strong, dim=1)

				loss_u = self.criterion_u(pred_u_augm_strong, labels_u)
				loss_u *= mask
				loss_u = self.reduce_fn(loss_u) * self.lambda_u

				loss_u.backward()
				self.optim.step()

				with torch.no_grad():
					self.recorder.add_scalar("train/loss_u", loss_u.item())
					self.recorder.add_scalar("train/labels_used", mask.mean().item())

					for metric_name, metric in self.metrics_u.items():
						score = metric(pred_u_augm_strong, labels_u)
						self.recorder.add_scalar(metric_name, score)

			self.printer.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch, self.name)

	def choose_loss(self) -> str:
		idx = self.categorical.sample().item()
		return ["s", "u"][idx]
