import torch

from metric_utils.metrics import Metrics

from ssl.mixmatch.loss import MixMatchLoss
from ssl.mixmatch.sharpen import sharpen
from ssl.mixmatch.warmup import WarmUp
from ssl.supervised.mixup import MixUp
from ssl.trainer_abc import TrainerABC
from mlu.utils.printers import ColumnPrinter
from mlu.utils.printers import PrinterABC
from ssl.utils.recorder.recorder_abc import RecorderABC
from ssl.utils.torch import merge_first_dimension, same_shuffle
from ssl.utils.types import IterableSized

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class MixMatchTrainer(TrainerABC):
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
		"""
		super().__init__()
		self.model = model
		self.activation = activation
		self.optim = optim
		self.criterion = criterion
		self.loader = loader
		self.metrics_s_mix = metrics_s_mix
		self.metrics_u_mix = metrics_u_mix
		self.recorder = recorder
		self.display = display
		self.device = device
		self.temperature = temperature
		self.lambda_s = lambda_s
		self.lambda_u = lambda_u

		self.mixup = MixUp(alpha, apply_max=True)
		self.warmup_lambda_u = WarmUp(lambda_u, warmup_nb_steps)

		self._mixup_lambda_s = None
		self._mixup_lambda_u = None

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

				for metric_name, metric in self.metrics_s_mix.items():
					_mean = metric(pred_s_mix, labels_s_mix)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				for metric_name, metric in self.metrics_u_mix.items():
					_mean = metric(pred_u_mix, labels_u_mix)
					self.recorder.add_point("train/{:s}".format(metric_name), metric.value.item())

				self.display.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch)
				self.warmup_lambda_u.step()

		self.recorder.end_record(epoch)

	def guess_label(self, batch_u_augm_weak_multiple: Tensor, temperature: float) -> Tensor:
		"""
			Guess the label of the unlabeled data by using the average prediction between K weakly augmented versions of the same unlabeled batch.

			:param batch_u_augm_weak_multiple: Tensor of shape (K, bsize, data size). Contains the K weakly augmented versions of the same batch.
			:param temperature: The temperature used for the sharpening operation that increase the higher probability.
		"""
		nb_augms = batch_u_augm_weak_multiple.shape[0]
		preds = [torch.zeros(0) for _ in range(nb_augms)]
		for k in range(nb_augms):
			logits = self.model(batch_u_augm_weak_multiple[k])
			preds[k] = self.activation(logits, dim=1)
		preds = torch.stack(preds)
		labels_u = preds.mean(dim=0)

		labels_u = sharpen(labels_u, temperature, dim=1)

		return labels_u

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

		batch_w = torch.cat((batch_s, batch_u_multiple))
		labels_w = torch.cat((labels_s, labels_u_multiple))

		# Shuffle batch and labels
		batch_w, labels_w = same_shuffle([batch_w, labels_w])

		len_s = len(batch_s)
		batch_s_mix, labels_s_mix = self.mixup(batch_s, batch_w[:len_s], labels_s, labels_w[:len_s])
		self._mixup_lambda_s = self.mixup.get_last_lambda()
		batch_u_mix, labels_u_mix = self.mixup(batch_u_multiple, batch_w[len_s:], labels_u_multiple, labels_w[len_s:])
		self._mixup_lambda_u = self.mixup.get_last_lambda()

		return batch_s_mix, batch_u_mix, labels_s_mix, labels_u_mix
