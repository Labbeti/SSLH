
import torch

from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Callable, List, Optional, Tuple

from mlu.metrics import MetricDict
from mlu.nn import CrossEntropyWithVectors
from sslh.experiments.mixmatch.mixmatch import MixMatch


class MixMatchNoMixUp(MixMatch):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Callable = Softmax(dim=-1),
		criterion_s: Module = CrossEntropyWithVectors(reduction="mean"),
		criterion_u: Module = CrossEntropyWithVectors(reduction="mean"),
		metric_dict_train_s: Optional[MetricDict] = None,
		metric_dict_train_u_pseudo: Optional[MetricDict] = None,
		metric_dict_val: Optional[MetricDict] = None,
		metric_dict_test: Optional[MetricDict] = None,
		log_on_epoch: bool = True,
		lambda_u: float = 1.0,
		nb_augms: int = 2,
		temperature: float = 0.5,
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
			lambda_u=lambda_u,
			nb_augms=nb_augms,
			temperature=temperature,
			alpha=0.0,
		)

	def training_step(self, batch: Tuple[Tuple[Tensor, Tensor], List[Tensor]], batch_idx: int) -> Tensor:
		(xs_weak, ys), xu_weak_lst = batch

		with torch.no_grad():
			# Guess pseudo-label "yu"
			yu = self.guess_label(xu_weak_lst)

			# Stack augmented "xu" variants to a single batch
			xu_weak_lst = torch.vstack(xu_weak_lst)
			yu_lst = yu.repeat([self.nb_augms] + [1] * (len(yu.shape) - 1))

		pred_xs_weak = self.activation(self.model(xs_weak))
		pred_xu_weak_lst = self.activation(self.model(xu_weak_lst))

		loss_s = self.criterion_s(pred_xs_weak, ys)
		loss_u = self.criterion_u(pred_xu_weak_lst, yu_lst)
		loss = loss_s + self.lambda_u * loss_u

		with torch.no_grad():
			self.log_dict({"train/loss": loss, "train/loss_s": loss_s, "train/loss_u": loss_u}, **self.log_params)

			scores_s = self.metric_dict_train_s(pred_xs_weak, ys)
			self.log_dict(scores_s, **self.log_params)

			scores_u = self.metric_dict_train_u_pseudo(pred_xu_weak_lst, yu_lst)
			self.log_dict(scores_u, **self.log_params)

		return loss
