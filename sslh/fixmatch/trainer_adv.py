
import math
import torch

from advertorch.attacks import GradientSignAttack

from mlu.metrics import Metric, CategoricalAccuracy
from mlu.nn import CrossEntropyWithVectors
from mlu.utils.misc import get_lr
from mlu.utils.printers import ColumnPrinter, PrinterABC

from sslh.fixmatch.loss import FixMatchLoss
from sslh.fixmatch.trainer import FixMatchTrainer
from sslh.utils.recorder.base import RecorderABC
from sslh.utils.types import IterableSized

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class FixMatchTrainerAdv(FixMatchTrainer):
	def __init__(
		self,
		model: Module,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics_s: Dict[str, Metric],
		metrics_u: Dict[str, Metric],
		recorder: RecorderABC,
		criterion: Callable = FixMatchLoss(),
		printer: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "train",
		threshold: float = 0.95,
		lambda_s: float = 1.0,
		lambda_u: float = 1.0,
		criterion_adv: Callable = CrossEntropyWithVectors(),
		epsilon_adv_weak: float = 1e-2,
		epsilon_adv_strong: float = 1e-1,
	):
		"""
			FixMatch with adversary instead of augmentations.

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
			:param criterion_adv: The criterion used for generating the adversary.
			:param epsilon_adv_weak: The epsilon value for weak FGSM attack. A higher value increase the distortion.
			:param epsilon_adv_strong: The epsilon value for weak FGSM attack. A higher value increase the distortion.
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

		self.metrics_u_adv = {"train/adv_acc": CategoricalAccuracy(dim=1)}

		criterion_adv_with_logits = lambda input_, target: criterion_adv(activation(input_, dim=1), target)
		self.adv_generator_weak = GradientSignAttack(
			predict=model,
			loss_fn=criterion_adv_with_logits,
			eps=epsilon_adv_weak,
			clip_min=-math.inf,
			clip_max=math.inf,
		)
		self.adv_generator_strong = GradientSignAttack(
			predict=model,
			loss_fn=criterion_adv_with_logits,
			eps=epsilon_adv_strong,
			clip_min=-math.inf,
			clip_max=math.inf,
		)

	def _train_impl(self, epoch: int):
		self.model.train()
		self.recorder.add_scalar("train/lr", get_lr(self.optim))

		for i, ((batch_s, labels_s), batch_u) in enumerate(self.loader):
			batch_s = batch_s.to(self.device).float()
			labels_s = labels_s.to(self.device).float()
			batch_u = batch_u.to(self.device).float()

			# Use guess u label with prediction of weak augmentation of u
			with torch.no_grad():
				labels_u, pred_u = self.guess_label(batch_u)
				mask = self.confidence_mask(pred_u, self.threshold, dim=1)

			batch_s_adv_weak = self.adv_generator_weak(batch_s, labels_s)
			batch_u_adv_strong = self.adv_generator_strong(batch_u, labels_u)

			self.optim.zero_grad()

			# Compute predictions
			logits_s_adv_weak = self.model(batch_s_adv_weak)
			logits_u_adv_strong = self.model(batch_u_adv_strong)

			pred_s_adv_weak = self.activation(logits_s_adv_weak, dim=1)
			pred_u_adv_strong = self.activation(logits_u_adv_strong, dim=1)

			# Update model
			loss, loss_s, loss_u = self.criterion(
				pred_s_adv_weak,
				pred_u_adv_strong,
				mask,
				labels_s,
				labels_u,
				lambda_s=self.lambda_s,
				lambda_u=self.lambda_u,
			)

			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.recorder.add_scalar("train/loss", loss.item())
				self.recorder.add_scalar("train/loss_s", loss_s.item())
				self.recorder.add_scalar("train/loss_u", loss_u.item())
				self.recorder.add_scalar("train/labels_used", mask.mean().item())

				for metric_name, metric in self.metrics_s.items():
					score = metric(pred_s_adv_weak, labels_s)
					self.recorder.add_scalar(metric_name, score)

				for metric_name, metric in self.metrics_u.items():
					score = metric(pred_u_adv_strong, labels_u)
					self.recorder.add_scalar(metric_name, score)

				for metric_name, metric in self.metrics_u_adv.items():
					score = metric(pred_u_adv_strong, labels_u)
					self.recorder.add_scalar(metric_name, score)

				self.printer.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch, self.name)
