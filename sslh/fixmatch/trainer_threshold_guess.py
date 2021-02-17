import torch

from mlu.metrics import Metric
from mlu.utils.printers import ColumnPrinter, PrinterABC

from sslh.fixmatch.loss import FixMatchLoss
from sslh.fixmatch.trainer import FixMatchTrainer
from sslh.utils.recorder.base import RecorderABC
from sslh.utils.types import IterableSized

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class FixMatchTrainerThresholdGuess(FixMatchTrainer):
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
			threshold_guess: float = 0.5,
	):
		"""
			FixMatch trainer.

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
			:param threshold_guess: TODO
		"""
		super().__init__(
			model=model,
			activation=activation,
			optim=optim,
			criterion=criterion,
			loader=loader,
			metrics_s=metrics_s,
			metrics_u=metrics_u,
			recorder=recorder,
			printer=printer,
			device=device,
			name=name,
			threshold=threshold,
			lambda_s=lambda_s,
			lambda_u=lambda_u,
		)
		self.threshold_guess = threshold_guess

	def guess_label(self, batch_u_augm_weak: Tensor) -> (Tensor, Tensor):
		logits_u_augm_weak = self.model(batch_u_augm_weak)
		pred_u_augm_weak = self.activation(logits_u_augm_weak, dim=1)

		labels_u = (pred_u_augm_weak >= self.threshold_guess).float()

		return labels_u, pred_u_augm_weak
