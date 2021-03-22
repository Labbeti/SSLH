
import torch

from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Callable, List, Optional, Tuple

from mlu.metrics import MetricDict
from mlu.nn.modules.loss import CrossEntropyWithVectors
from sslh.experiments.mixmatch.mixmatch import MixMatch
from sslh.transforms import get_self_transform
from sslh.utils.average_pred import AveragePred


class ReMixMatch(MixMatch):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Callable = Softmax(dim=-1),
		activation_r: Callable = Softmax(dim=-1),
		criterion_s: Module = CrossEntropyWithVectors(reduction="mean"),
		criterion_u: Module = CrossEntropyWithVectors(reduction="mean"),
		criterion_u1: Module = CrossEntropyWithVectors(reduction="mean"),
		criterion_r: Module = CrossEntropyWithVectors(reduction="mean"),
		metric_dict_train_s: Optional[MetricDict] = None,
		metric_dict_train_u_pseudo: Optional[MetricDict] = None,
		metric_dict_val: Optional[MetricDict] = None,
		metric_dict_test: Optional[MetricDict] = None,
		metric_dict_train_r: Optional[MetricDict] = None,
		log_on_epoch: bool = True,
		self_transform: Callable[[Tensor], Tuple[Tensor, Tensor]] = get_self_transform("cifar10"),
		lambda_u: float = 1.5,
		lambda_u1: float = 0.5,
		lambda_r: float = 0.5,
		nb_augms: int = 2,
		temperature: float = 0.5,
		alpha: float = 0.75,
		history: int = 128,
		check_model: bool = True,
	):
		if check_model and (not hasattr(model, "forward_rot") or not callable(model.forward_rot)):
			raise RuntimeError(
				f"Model '{model.__class__.__name__}' does not have a method 'forward_rot()' method for predict a "
				f"random rotation. Maybe use '{model.__class__.__name__}Rot' ?"
			)

		if metric_dict_train_r is None:
			metric_dict_train_r = MetricDict()

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
			alpha=alpha,
		)
		self.activation_r = activation_r
		self.criterion_u1 = criterion_u1
		self.criterion_r = criterion_r
		self.metric_dict_train_r = metric_dict_train_r
		self.self_transform = self_transform

		self.lambda_u1 = lambda_u1
		self.lambda_r = lambda_r
		self.history = history

		self.average_pred_s = AveragePred(history)
		self.average_pred_u = AveragePred(history)

		self.save_hyperparameters("lambda_u1", "lambda_r", "history")

	def training_step(self, batch: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]], batch_idx: int) -> Tensor:
		(xs_strong, ys), (xu_weak, xu_strong_lst) = batch

		with torch.no_grad():
			pred_xu_weak = self.activation(self.model(xu_weak))

			self.average_pred_s.add_pred(ys)
			self.average_pred_u.add_pred(pred_xu_weak)

			# Guess pseudo-label "yu"
			yu = pred_xu_weak * self.average_pred_s.get_mean() / self.average_pred_u.get_mean()
			yu = yu / yu.norm(p=1, dim=-1, keepdim=True)
			yu = self.sharpen(yu)
			yu_lst = yu.repeat([self.nb_augms + 1] + [1] * (len(yu.shape) - 1))

			xu_lst = [xu_weak] + xu_strong_lst
			xs_strong_mix, xu_weak_and_strong_mix, ys_mix, yu_mix = self.mixmatch(xs_strong, xu_lst, ys, yu_lst)

			xu1_strong = xu_strong_lst[0].clone()
			yu1 = yu

			xu1_strong_rotated, yu1_r = self.self_transform(xu1_strong)

		pred_xs_mix = self.activation(self.model(xs_strong_mix))
		pred_xu_mix = self.activation(self.model(xu_weak_and_strong_mix))
		pred_xu1 = self.activation(self.model(xu1_strong))
		pred_r = self.activation_r(self.model.forward_rot(xu1_strong_rotated))

		loss_s = self.criterion_s(pred_xs_mix, ys_mix)
		loss_u = self.criterion_u(pred_xu_mix, yu_mix)
		loss_u1 = self.criterion_u1(pred_xu1, yu1)
		loss_r = self.criterion_r(pred_r, yu1_r)

		loss = loss_s + self.lambda_u * loss_u + self.lambda_u1 * loss_u1 + self.lambda_r * loss_r

		with torch.no_grad():
			self.log_dict({"train/loss": loss, "train/loss_s": loss_s, "train/loss_u": loss_u}, **self.log_params)

			pred_xs_strong = self.activation(self.model(xs_strong))
			scores_s = self.metric_dict_train_s(pred_xs_strong, ys)
			self.log_dict(scores_s, **self.log_params)

			pred_xu_strong_lst = self.activation(self.model(torch.vstack(xu_lst)))
			scores_u = self.metric_dict_train_u_pseudo(pred_xu_strong_lst, yu_lst)
			self.log_dict(scores_u, **self.log_params)

			scores_u1 = self.metric_dict_train_u_pseudo(pred_xu1, yu1)
			self.log_dict(scores_u1, **self.log_params)

			scores_r = self.metric_dict_train_r(pred_r, yu1_r)
			self.log_dict(scores_r, **self.log_params)

		return loss
