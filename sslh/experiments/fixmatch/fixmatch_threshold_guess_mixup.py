
import torch

from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Callable, Optional, Tuple

from mlu.metrics import MetricDict
from mlu.nn import CrossEntropyWithVectors, OneHot
from sslh.experiments.fixmatch.fixmatch_mixup import FixMatchMixUp


class FixMatchThresholdGuessMixUp(FixMatchMixUp):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Callable = Softmax(dim=-1),
		criterion_s: Module = CrossEntropyWithVectors(reduction="none"),
		criterion_u: Module = CrossEntropyWithVectors(reduction="none"),
		metric_dict_train_s: Optional[MetricDict] = None,
		metric_dict_train_u_pseudo: Optional[MetricDict] = None,
		metric_dict_val: Optional[MetricDict] = None,
		metric_dict_test: Optional[MetricDict] = None,
		log_on_epoch: bool = True,
		target_transform: Callable = OneHot(num_classes=10),
		lambda_u: float = 1.0,
		threshold: float = 0.0,
		threshold_guess: float = 0.75,
		alpha: float = 0.75,
	):
		super().__init__(
			model=model,
			optimizer=optimizer,
			activation=activation,
			criterion_s=criterion_s,
			criterion_u=criterion_u,
			metric_dict_train_s=metric_dict_train_s,
			metric_dict_train_u_pseudo=metric_dict_train_u_pseudo,
			metric_dict_val=metric_dict_val,
			metric_dict_test=metric_dict_test,
			log_on_epoch=log_on_epoch,
			target_transform=target_transform,
			lambda_u=lambda_u,
			threshold=threshold,
			alpha=alpha,
		)
		self.threshold_guess = threshold_guess

		self.save_hyperparameters("threshold_guess", "alpha")

	def guess_label_and_mask(self, xu_weak: Tensor) -> Tuple[Tensor, Tensor]:
		with torch.no_grad():
			pred_xu_weak = self.model(xu_weak)
			yu = pred_xu_weak.ge(self.threshold_guess).to(pred_xu_weak.dtype)
			probabilities_max, _ = pred_xu_weak.max(dim=-1)
			mask = probabilities_max.ge(self.threshold).to(pred_xu_weak.dtype)
			return yu, mask
