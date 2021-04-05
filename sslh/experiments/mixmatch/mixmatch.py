
import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Tuple

from mlu.metrics import MetricDict
from mlu.nn.modules.loss import CrossEntropyWithVectors
from sslh.transforms.augments.mixup import MixUpModule


class MixMatch(LightningModule):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Module = Softmax(dim=-1),
		criterion_s: Module = CrossEntropyWithVectors(reduction="mean"),
		criterion_u: Module = CrossEntropyWithVectors(reduction="mean"),
		lambda_u: float = 1.0,
		nb_augms: int = 2,
		temperature: float = 0.5,
		alpha: float = 0.75,
		metric_dict_train_s: Optional[MetricDict] = None,
		metric_dict_train_u_pseudo: Optional[MetricDict] = None,
		metric_dict_val: Optional[MetricDict] = None,
		metric_dict_test: Optional[MetricDict] = None,
		log_on_epoch: bool = True,
	):
		if metric_dict_train_s is None:
			metric_dict_train_s = MetricDict()
		if metric_dict_train_u_pseudo is None:
			metric_dict_train_u_pseudo = MetricDict()
		if metric_dict_val is None:
			metric_dict_val = MetricDict()
		if metric_dict_test is None:
			metric_dict_test = MetricDict()

		super().__init__()
		self.model = model
		self.activation = activation
		self.optimizer = optimizer
		self.criterion_s = criterion_s
		self.criterion_u = criterion_u
		self.lambda_u = lambda_u
		self.nb_augms = nb_augms
		self.temperature = temperature
		self.alpha = alpha
		self.metric_dict_train_s = metric_dict_train_s
		self.metric_dict_train_u_pseudo = metric_dict_train_u_pseudo
		self.metric_dict_val = metric_dict_val
		self.metric_dict_test = metric_dict_test

		self.mixup = MixUpModule(alpha=alpha, apply_max=True)

		self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)
		self.save_hyperparameters({
			"experiment": self.__class__.__name__,
			"model": model.__class__.__name__,
			"optimizer": optimizer.__class__.__name__,
			"activation": activation.__class__.__name__,
			"criterion_s": criterion_s.__class__.__name__,
			"criterion_u": criterion_u.__class__.__name__,
			"lambda_u": lambda_u,
			"nb_augms": nb_augms,
			"temperature": temperature,
			"alpha": alpha,
		})

	def training_step(self, batch: Tuple[Tuple[Tensor, Tensor], List[Tensor]], batch_idx: int) -> Tensor:
		(xs_weak, ys), xu_weak_lst = batch

		with torch.no_grad():
			# Guess pseudo-label "yu" and repeat
			yu = self.guess_label(xu_weak_lst)
			yu_lst = yu.repeat([self.nb_augms] + [1] * (len(yu.shape) - 1))

			xs_weak_mix, xu_weak_mix, ys_mix, yu_mix = self.mixmatch(xs_weak, xu_weak_lst, ys, yu_lst)

		pred_xs_mix = self.activation(self.model(xs_weak_mix))
		pred_xu_mix = self.activation(self.model(xu_weak_mix))

		loss_s = self.criterion_s(pred_xs_mix, ys_mix)
		loss_u = self.criterion_u(pred_xu_mix, yu_mix)
		loss = loss_s + self.lambda_u * loss_u

		with torch.no_grad():
			self.log_dict({"train/loss": loss, "train/loss_s": loss_s, "train/loss_u": loss_u}, **self.log_params)

			pred_xs_weak = self.activation(self.model(xs_weak))
			scores_s = self.metric_dict_train_s(pred_xs_weak, ys)
			self.log_dict(scores_s, **self.log_params)

			pred_xu_weak_lst = self.activation(self.model(torch.vstack(xu_weak_lst)))
			scores_u = self.metric_dict_train_u_pseudo(pred_xu_weak_lst, yu_lst)
			self.log_dict(scores_u, **self.log_params)

		return loss

	def mixmatch(self, xs_weak: Tensor, xu_weak_lst: List[Tensor], ys: Tensor, yu_lst: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
		# Stack augmented "xu" variants to a single batch
		xu_weak_lst = torch.vstack(xu_weak_lst)

		# Prepare W
		xw = torch.cat((xs_weak, xu_weak_lst))
		yw = torch.cat((ys, yu_lst))

		# Shuffle W
		indices = torch.randperm(len(xw))
		xw, yw = xw[indices], yw[indices]

		# Apply MixUp twice
		bsize_s = len(xs_weak)
		xs_weak_mix, ys_mix = self.mixup(xs_weak, xw[:bsize_s], ys, yw[:bsize_s])
		xu_weak_mix, yu_mix = self.mixup(xu_weak_lst, xw[bsize_s:], yu_lst, yw[bsize_s:])

		return xs_weak_mix, xu_weak_mix, ys_mix, yu_mix

	def guess_label(self, xu_weak_lst: List[Tensor]) -> Tensor:
		pred_xu_weak_lst = self.activation(self.model(xu_weak_lst[0]))
		for xu_weak_lst in xu_weak_lst[1:]:
			pred_xu_weak_lst += self.activation(self.model(xu_weak_lst))
		pred_xu_weak_lst /= self.nb_augms
		yu = self.sharpen(pred_xu_weak_lst)
		return yu

	def sharpen(self, pred: Tensor) -> Tensor:
		pred = pred ** (1.0 / self.temperature)
		pred = pred / pred.norm(p=1, dim=-1, keepdim=True)
		return pred

	def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_xs = self.activation(self.model(xs))
		self.log_dict(self.metric_dict_val(pred_xs, ys), on_epoch=True, on_step=False)

	def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_xs = self.activation(self.model(xs))
		self.log_dict(self.metric_dict_test(pred_xs, ys), on_epoch=True, on_step=False)

	def forward(self, x: Tensor) -> Tensor:
		return self.activation(self.model(x))

	def configure_optimizers(self) -> Optimizer:
		return self.optimizer
