
import torch

from mlu.metrics import Metric
from mlu.utils.printers import ColumnPrinter, PrinterABC
from mlu.utils.misc import get_lr

from sslh.mixmatch.loss import MixMatchLossNoLabelMix
from sslh.mixmatch.trainer import MixMatchTrainer
from sslh.utils.recorder.base import RecorderABC
from sslh.utils.torch import collapse_first_dimension
from sslh.utils.types import IterableSized

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
		metrics_s_mix: Dict[str, Metric],
		metrics_u_mix: Dict[str, Metric],
		recorder: RecorderABC,
		criterion: Callable = MixMatchLossNoLabelMix(),
		printer: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "train",
		temperature: float = 0.5,
		alpha: float = 0.75,
		lambda_s: float = 1.0,
		lambda_u: float = 1.0,
		warmup_nb_steps: int = 16000,
		use_warmup_by_iteration: bool = True,
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
			:param criterion: The loss function. (default: MixMatchLoss())
			:param printer: The object used to print values during training. (default: ColumnPrinter())
			:param device: The Pytorch device used for tensors. (default: torch.device('cuda'))
			:param name: The name of the training. (default: 'train')
			:param temperature: The temperature used in sharpening function for post-process labels. (default: 0.5)
			:param alpha: The alpha hyperparameter for MixUp. (default: 0.75)
			:param lambda_s: The coefficient of labeled loss component. (default: 1.0)
			:param lambda_u: The coefficient of unlabeled loss component. (default: 1.0)
			:param warmup_nb_steps: The number of steps used to increase linearly the lambda_u hyperparameter. (default: 16000)
			:param use_warmup_by_iteration: Activate WarmUp on lambda_u hyperparameter. (default: True)
		"""
		super().__init__(
			model=model,
			activation=activation,
			optim=optim,
			loader=loader,
			metrics_s_mix=metrics_s_mix,
			metrics_u_mix=metrics_u_mix,
			recorder=recorder,
			criterion=criterion,
			printer=printer,
			device=device,
			name=name,
			temperature=temperature,
			alpha=alpha,
			lambda_s=lambda_s,
			lambda_u=lambda_u,
			warmup_nb_steps=warmup_nb_steps,
			use_warmup_by_iteration=use_warmup_by_iteration,
		)
		self._prev_labels_s_shuffle = None
		self._prev_labels_u_shuffle = None
		self._prev_labels_u_multiple = None
		self._mixup_lambda_s = None
		self._mixup_lambda_u = None

	def _train_impl(self, epoch: int):
		self.model.train()
		self.recorder.add_scalar("train/lr", get_lr(self.optim))

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
				lambda_u=self.lambda_u,
				lambda_s_shuffle=self._mixup_lambda_s,
				lambda_u_shuffle=self._mixup_lambda_u,
			)
			loss.backward()
			self.optim.step()

			# Compute metrics
			with torch.no_grad():
				self.recorder.add_scalar("train/loss", loss.item())
				self.recorder.add_scalar("train/loss_s", loss_s.item())
				self.recorder.add_scalar("train/loss_u", loss_u.item())
				self.recorder.add_scalar("train/lambda_u", self.lambda_u)

				for metric_name, metric in self.metrics_s_mix.items():
					score = metric(pred_s_mix, labels_s_mix)
					self.recorder.add_scalar(metric_name, score)

				for metric_name, metric in self.metrics_u_mix.items():
					score = metric(pred_u_mix, labels_u_mix)
					self.recorder.add_scalar(metric_name, score)

				self.printer.print_current_values(self.recorder.get_current_means(), i, len(self.loader), epoch, self.name)
				if self.use_warmup_by_iteration:
					self.warmup_lambda_u.step()

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
		batch_u_multiple = collapse_first_dimension(batch_u_multiple)
		self._prev_labels_u_multiple = labels_u_multiple

		batch_w = torch.cat((batch_s, batch_u_multiple))
		labels_w = torch.cat((labels_s, labels_u_multiple))

		# Shuffle batch and labels
		indices = torch.randperm(batch_w.shape[0])
		batch_w, labels_w = batch_w[indices], labels_w[indices]

		len_s = len(batch_s)
		batch_s_mix, labels_s_mix = self.mixup(batch_s, batch_w[:len_s], labels_s, labels_w[:len_s])
		self._mixup_lambda_s = self.mixup.get_last_lambda()
		self._prev_labels_s_shuffle = labels_w[:len_s]

		batch_u_mix, labels_u_mix = self.mixup(batch_u_multiple, batch_w[len_s:], labels_u_multiple, labels_w[len_s:])
		self._mixup_lambda_u = self.mixup.get_last_lambda()
		self._prev_labels_u_shuffle = labels_w[len_s:]

		return batch_s_mix, batch_u_mix, labels_s_mix, labels_u_mix
