
import torch

from metric_utils.metrics import Metrics

from ssl.mixmatch.loss import MixMatchLossNoLabelMix
from ssl.mixmatch.trainer import MixMatchTrainer
from mlu.utils.printers import ColumnPrinter
from mlu.utils.printers import PrinterABC
from ssl.utils.recorder.recorder_abc import RecorderABC
from ssl.utils.torch import merge_first_dimension, same_shuffle
from ssl.utils.types import IterableSized

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class MixMatchTrainerNoLabelMix(MixMatchTrainer):
	def __init__(
		self,
		model: Module,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics_s_mix: Dict[str, Metrics],
		metrics_u_mix: Dict[str, Metrics],
		recorder: RecorderABC,
		criterion: Callable = MixMatchLossNoLabelMix(),
		display: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		temperature: float = 0.5,
		alpha: float = 0.75,
		lambda_s: float = 1.0,
		lambda_u: float = 1.0,
		warmup_nb_steps: int = 16000,
	):
		"""
			MixMatch trainer without mixing labels. The criterion must be like MixMatchLossNoLabelMix.

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
		"""
		super().__init__(
			model, activation, optim, loader, metrics_s_mix, metrics_u_mix, recorder, criterion, display, device, temperature, alpha, lambda_s, lambda_u, warmup_nb_steps
		)
		self._prev_labels_s_shuffle = None
		self._prev_labels_u_shuffle = None
		self._prev_labels_u_multiple = None

	def _train_impl(self, epoch: int):
		self.model.train()
		for metrics in (self.metrics_s_mix.values(), self.metrics_u_mix.values()):
			for metric in metrics:
				metric.reset()

		self.recorder.start_record(epoch)

		for i, ((batch_s_augm_weak, labels_s), batch_u_augm_weak_multiple) in enumerate(self.loader):
			batch_s_augm_weak = batch_s_augm_weak.to(self.device).float()
			labels_s = labels_s.to(self.device).float()
			batch_u_augm_weak_multiple = torch.stack(batch_u_augm_weak_multiple).to(self.device).float()

			with torch.no_grad():
				labels_u = self.guess_label(batch_u_augm_weak_multiple, self.temperature)
				batch_s_mix, batch_u_mix, labels_s_mix, labels_u_mix = self.mixmatch(
					batch_s_augm_weak, batch_u_augm_weak_multiple, labels_s, labels_u)

			self.optim.zero_grad()

			logits_s_mix = self.model(batch_s_mix)
			logits_u_mix = self.model(batch_u_mix)

			pred_s_mix = self.activation(logits_s_mix, dim=1)
			pred_u_mix = self.activation(logits_u_mix, dim=1)

			loss, loss_s, loss_u = self.criterion(
				pred_s_mix,
				pred_u_mix,
				labels_s,
				self._prev_labels_u_multiple,
				self._prev_labels_s_shuffle,
				self._prev_labels_u_shuffle,
				lambda_s=self.lambda_s,
				lambda_u=self.warmup_lambda_u.get_value(),
				lambda_s_shuffle=self._mixup_lambda_s,
				lambda_u_shuffle=self._mixup_lambda_u,
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

				for metric_name, metric in self.metrics_s_mix.items():
					_mean = metric(pred_s_mix, labels_s_mix)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				for metric_name, metric in self.metrics_u_mix.items():
					_mean = metric(pred_u_mix, labels_u_mix)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				self.display.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch)
				self.warmup_lambda_u.step()

		self.recorder.end_record(epoch)

	def mixmatch(self, batch_s: Tensor, batch_u_multiple: Tensor, labels_s: Tensor, labels_u: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		"""
			:param batch_s: Labeled batch of shape (batch_size, ...)
			:param batch_u_multiple: Unlabeled batch of shape (nb_augms, batch_size, ...)
			:param labels_s: Label of s of shape (batch_size, nb_classes)
			:param labels_u: Label of u of shape (batch_size, nb_classes)
		"""
		nb_augms = batch_u_multiple.shape[0]
		repeated_size = [nb_augms] + [1] * (len(labels_u.shape) - 1)
		labels_u_multiple = labels_u.repeat(repeated_size)
		batch_u_multiple = merge_first_dimension(batch_u_multiple)
		self._prev_labels_u_multiple = labels_u_multiple

		batch_w = torch.cat((batch_s, batch_u_multiple))
		labels_w = torch.cat((labels_s, labels_u_multiple))

		# Shuffle batch and labels
		batch_w, labels_w = same_shuffle([batch_w, labels_w])

		len_s = len(batch_s)
		batch_s_mix, labels_s_mix = self.mixup(batch_s, batch_w[:len_s], labels_s, labels_w[:len_s])
		self._mixup_lambda_s = self.mixup.get_last_lambda()
		self._prev_labels_s_shuffle = labels_w[:len_s]

		batch_u_mix, labels_u_mix = self.mixup(batch_u_multiple, batch_w[len_s:], labels_u_multiple, labels_w[len_s:])
		self._mixup_lambda_u = self.mixup.get_last_lambda()
		self._prev_labels_u_shuffle = labels_w[len_s:]

		return batch_s_mix, batch_u_mix, labels_s_mix, labels_u_mix
