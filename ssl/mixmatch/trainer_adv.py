
import math
import torch

from advertorch.attacks import GradientSignAttack

from metric_utils.metrics import Metrics

from ssl.mixmatch.loss import MixMatchLoss
from ssl.mixmatch.trainer import MixMatchTrainer
from mlu.utils.printers import ColumnPrinter
from mlu.utils.printers import PrinterABC
from ssl.utils.other_metrics import CategoricalAccuracyOnehot
from ssl.utils.recorder.recorder_abc import RecorderABC
from mlu.nn import CrossEntropyWithVectors
from ssl.utils.types import IterableSized

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class MixMatchTrainerAdv(MixMatchTrainer):
	"""
		MixMatch with adversarial example used instead of augmentations.
	"""
	def __init__(
		self,
		model: Module,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics_s_mix: Dict[str, Metrics],
		metrics_u_mix: Dict[str, Metrics],
		recorder: RecorderABC,
		criterion: Callable = MixMatchLoss(),
		display: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		temperature: float = 0.5,
		alpha: float = 0.75,
		lambda_s: float = 1.0,
		lambda_u: float = 1.0,
		warmup_nb_steps: int = 16000,
		criterion_adv: Callable = CrossEntropyWithVectors(),
		epsilon_adv: float = 1e-2,
	):
		"""
			MixMatch trainer.

			:param model: The pytorch model to train.
			:param activation: The activation function of the model. (Inputs: (x: Tensor, dim: int)).
			:param optim: The optimizer used to update the model.
			:param loader: The dataloader used to load ((batch_s_weak, labels_s), (batch_u_weak, batch_u_strong))
			:param metrics_s_mix: Metrics used during training on mixed prediction labeled and labels.
			:param metrics_u_mix: Metrics used during training on mixed prediction unlabeled and labels.
			:param recorder: The recorder used to store metrics.
			:param criterion: The loss function.
			:param display: The object used to print values during training.
			:param device: The Pytorch device used for tensors.
			:param temperature: The temperature used in sharpening function for Pseudo-labeling.
			:param alpha: The alpha hyperparameter for MixUp.
			:param lambda_s: The coefficient of labeled loss component.
			:param lambda_u: The coefficient of unlabeled loss component.
			:param warmup_nb_steps: The number of steps used to increase linearly the lambda_u hyperparameter.
			:param criterion_adv: The criterion used for generate adversarial examples.
			:param epsilon_adv: The epsilon used for Gradient Sign Attack generator.
		"""
		super().__init__(
			model, activation, optim, loader, metrics_s_mix, metrics_u_mix, recorder, criterion, display, device, temperature, alpha, lambda_s, lambda_u, warmup_nb_steps
		)
		self.adv_acc_metric = CategoricalAccuracyOnehot(dim=1)
		self.adv_generator = GradientSignAttack(
			predict=model,
			loss_fn=lambda input_, target: criterion_adv(activation(input_, dim=1), target),
			eps=epsilon_adv,
			clip_min=-math.inf,
			clip_max=math.inf,
		)

	def _train_impl(self, epoch: int):
		self.model.train()
		for metrics in (self.metrics_s_mix.values(), self.metrics_u_mix.values()):
			for metric in metrics:
				metric.reset()

		self.recorder.start_record(epoch)

		for i, ((batch_s, labels_s), batch_u_multiple) in enumerate(self.loader):
			batch_s = batch_s.to(self.device).float()
			labels_s = labels_s.to(self.device).float()
			batch_u_multiple = torch.stack(batch_u_multiple).to(self.device).float()

			with torch.no_grad():
				labels_u = self.guess_label(batch_u_multiple, self.temperature)

			batch_s_adv = self.adv_generator(batch_s, labels_s)

			batch_u_multiple_adv = torch.zeros_like(batch_u_multiple)
			for j in range(batch_u_multiple.shape[0]):
				batch_u_multiple_adv[j] = self.adv_generator(batch_u_multiple[j], labels_u)

			with torch.no_grad():
				batch_s_mix, batch_u_mix, labels_s_mix, labels_u_mix = self.mixmatch(
					batch_s_adv,
					batch_u_multiple_adv,
					labels_s,
					labels_u
				)

			self.optim.zero_grad()

			logits_s_mix = self.model(batch_s_mix)
			logits_u_mix = self.model(batch_u_mix)

			pred_s_mix = self.activation(logits_s_mix, dim=1)
			pred_u_mix = self.activation(logits_u_mix, dim=1)

			loss, loss_s, loss_u = self.criterion(
				pred_s_mix,
				pred_u_mix,
				labels_s_mix,
				labels_u_mix,
				lambda_s=self.lambda_s,
				lambda_u=self.warmup_lambda_u.get_value()
			)
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.recorder.add_point("train/loss", loss.item())
				self.recorder.add_point("train/loss_s", loss_s.item())
				self.recorder.add_point("train/loss_u", loss_u.item())
				self.recorder.set_point("train/lambda_u", self.warmup_lambda_u.get_value())
				self.recorder.set_point("train/mixup_lambda", self._mixup_lambda_s)
				self.recorder.set_point("train/mixup_lambda", self._mixup_lambda_u)

				for batch_u_adv in batch_u_multiple_adv:
					pred_u_adv = self.activation(self.model(batch_u_adv), dim=1)
					self.adv_acc_metric(pred_u_adv, labels_u)
					self.recorder.add_point("train/adv_acc", self.adv_acc_metric.value.item())

				for metric_name, metric in self.metrics_s_mix.items():
					_mean = metric(pred_s_mix, labels_s_mix)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				for metric_name, metric in self.metrics_u_mix.items():
					_mean = metric(pred_u_mix, labels_u_mix)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				self.display.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch)
				self.warmup_lambda_u.step()

		self.recorder.end_record(epoch)
