import torch

from mlu.metrics import Metric
from mlu.utils.printers import ColumnPrinter, PrinterABC

from sslh.mixmatch.loss import MixMatchLoss
from sslh.mixmatch.trainer import MixMatchTrainer
from sslh.utils.recorder.base import RecorderABC
from sslh.utils.torch import collapse_first_dimension
from sslh.utils.types import IterableSized

from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class MixMatchTrainerNoMixUp(MixMatchTrainer):
	def __init__(
		self,
		model: Module,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics_s_mix: Dict[str, Metric],
		metrics_u_mix: Dict[str, Metric],
		recorder: RecorderABC,
		criterion: Callable = MixMatchLoss(),
		printer: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "train",
		temperature: float = 0.5,
		lambda_s: float = 1.0,
		lambda_u: float = 1.0,
		warmup_nb_steps: int = 16000,
		use_warmup_by_iteration: bool = True,
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
			:param criterion: The loss function. (default: MixMatchLoss())
			:param printer: The object used to print values during training. (default: ColumnPrinter())
			:param device: The Pytorch device used for tensors. (default: torch.device('cuda'))
			:param name: The name of the training. (default: 'train')
			:param temperature: The temperature used in sharpening function for post-process labels. (default: 0.5)
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
			alpha=0.0,
			lambda_s=lambda_s,
			lambda_u=lambda_u,
			warmup_nb_steps=warmup_nb_steps,
			use_warmup_by_iteration=use_warmup_by_iteration,
		)

	def mixmatch(self, batch_s: Tensor, batch_u_multiple: Tensor, labels_s: Tensor, labels_u: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		"""
			MixMatch part WITHOUT MixUp between labeled and unlabeled data.

			:param batch_s: Labeled batch of shape (batch_size, ...)
			:param batch_u_multiple: Unlabeled batch of shape (nb_augms, batch_size, ...)
			:param labels_s: Label of s of shape (batch_size, nb_classes)
			:param labels_u: Label of u of shape (batch_size, nb_classes)
		"""
		nb_augms = batch_u_multiple.shape[0]
		repeated_size = [nb_augms] + [1] * (len(labels_u.shape) - 1)
		labels_u_multiple = labels_u.repeat(repeated_size)
		batch_u_multiple = collapse_first_dimension(batch_u_multiple)

		return batch_s, batch_u_multiple, labels_s, labels_u_multiple
