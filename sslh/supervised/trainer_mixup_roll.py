import torch

from mlu.metrics import Metric
from mlu.utils.printers import ColumnPrinter, PrinterABC
from mlu.utils.misc import get_lr

from sslh.augments.mixup_roll import MixUpRoll
from sslh.supervised.loss import MixUpLoss
from sslh.supervised.trainer_mixup import SupervisedTrainerMixUp
from sslh.utils.recorder.base import RecorderABC
from sslh.utils.types import IterableSized

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from typing import Callable, Dict


class SupervisedMixUpRollTrainer(SupervisedTrainerMixUp):
	def __init__(
		self,
		model: Module,
		activation: Callable,
		optim: Optimizer,
		loader: IterableSized,
		metrics: Dict[str, Metric],
		recorder: RecorderABC,
		criterion: Callable = MixUpLoss(),
		printer: PrinterABC = ColumnPrinter(),
		device: torch.device = torch.device("cuda"),
		name: str = "train",
		alpha: float = 0.4,
		apply_max: bool = False,
		dim_rool: int = -1,
	):
		"""
			Supervised trainer.

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
			:param alpha: The MixUp alpha hyperparameter used for generating the mixup_lambda coefficient for mixing
				batches and labels. (default: 1.0)
			:param apply_max: Apply Max on lambda MixUp for generate asymmetric mixes.
			:param dim_rool: The dimension index to roll.
		"""
		super().__init__(
			model,
			activation,
			optim,
			loader,
			metrics,
			recorder,
			criterion,
			printer,
			device,
			name,
			alpha,
			apply_max,
		)

		self.mixup = MixUpRoll(dim_roll=dim_rool, alpha=alpha, apply_max=apply_max)
