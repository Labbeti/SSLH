
import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Dict, List, Optional, Tuple

from mlu.nn import ForwardDictAffix
from mlu.nn import CrossEntropyWithVectors
from sslh.transforms.augments.mixup import MixUpModule


class MixMatch(LightningModule):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Module = Softmax(dim=-1),
		criterion_s: Module = CrossEntropyWithVectors(reduction='mean'),
		criterion_u: Module = CrossEntropyWithVectors(reduction='mean'),
		lambda_u: float = 1.0,
		n_augms: int = 2,
		temperature: float = 0.5,
		alpha: float = 0.75,
		train_metrics: Optional[Dict[str, Module]] = None,
		val_metrics: Optional[Dict[str, Module]] = None,
		log_on_epoch: bool = True,
	):
		"""
			MixMatch (MM) LightningModule.

			:param model: The PyTorch Module to train.
				The forward() must return logits for classify the data.
			:param optimizer: The PyTorch optimizer to use.
			:param activation: The activation function of the model.
				(default: Softmax(dim=-1))
			:param criterion_s: The loss component 'L_s' of MM.
				(default: CrossEntropyWithVectors())
			:param criterion_u: The loss component 'L_u' of MM.
				(default: CrossEntropyWithVectors())
			:param lambda_u: The coefficient of the 'L_u' component. (default: 1.0)
			:param n_augms: The number of strong augmentations applied. (default: 2)
			:param temperature: The temperature applied by the sharpen function.
				A lower temperature make the pseudo-label produced more 'one-hot'.
				(default: 0.5)
			:param alpha: The mixup alpha parameter. A higher value means a stronger mix between labeled and unlabeled data.
				(default: 0.75)
			:param train_metrics: An optional dictionary of metrics modules for training.
				(default: None)
			:param val_metrics: An optional dictionary of metrics modules for validation.
				(default: None)
			:param log_on_epoch: If True, log only the epoch means of each train metric score.
				(default: True)
		"""
		super().__init__()
		self.model = model
		self.activation = activation
		self.optimizer = optimizer
		self.criterion_s = criterion_s
		self.criterion_u = criterion_u
		self.lambda_u = lambda_u
		self.n_augms = n_augms
		self.temperature = temperature
		self.alpha = alpha

		self.metric_dict_train_s = ForwardDictAffix(train_metrics, prefix='train/', suffix='_s')
		self.metric_dict_train_u_pseudo = ForwardDictAffix(train_metrics, prefix='train/', suffix='_u')
		self.metric_dict_val = ForwardDictAffix(val_metrics, prefix='val/')
		self.metric_dict_test = ForwardDictAffix(val_metrics, prefix='test/')

		self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)
		self.mixup = MixUpModule(alpha=alpha, apply_max=True)

		self.save_hyperparameters({
			'experiment': self.__class__.__name__,
			'model': model.__class__.__name__,
			'optimizer': optimizer.__class__.__name__,
			'activation': activation.__class__.__name__,
			'criterion_s': criterion_s.__class__.__name__,
			'criterion_u': criterion_u.__class__.__name__,
			'lambda_u': lambda_u,
			'n_augms': n_augms,
			'temperature': temperature,
			'alpha': alpha,
		})

	def training_step(self, batch: Tuple[Tuple[Tensor, Tensor], List[Tensor]], batch_idx: int) -> Tensor:
		(xs_weak, ys), xu_weak_lst = batch

		with torch.no_grad():
			# Guess pseudo-label 'yu' and repeat
			yu = self.guess_label(xu_weak_lst)
			yu_lst = yu.repeat([self.n_augms] + [1] * (len(yu.shape) - 1))

			# Stack augmented 'xu' variants to a single batch
			xu_weak_lst = torch.vstack(xu_weak_lst)

			xs_weak_mix, xu_weak_mix, ys_mix, yu_mix = self.mixmatch(xs_weak, xu_weak_lst, ys, yu_lst)

		pred_xs_mix = self.activation(self.model(xs_weak_mix))
		pred_xu_mix = self.activation(self.model(xu_weak_mix))

		loss_s = self.criterion_s(pred_xs_mix, ys_mix)
		loss_u = self.criterion_u(pred_xu_mix, yu_mix)
		loss = loss_s + self.lambda_u * loss_u

		with torch.no_grad():
			scores = {'train/loss': loss, 'train/loss_s': loss_s, 'train/loss_u': loss_u}
			scores = {k: v.cpu() for k, v in scores.items()}
			self.log_dict(scores, **self.log_params)

			pred_xs_weak = self.activation(self.model(xs_weak))
			scores_s = self.metric_dict_train_s(pred_xs_weak, ys)
			self.log_dict(scores_s, **self.log_params)

			pred_xu_weak_lst = self.activation(self.model(xu_weak_lst))
			scores_u = self.metric_dict_train_u_pseudo(pred_xu_weak_lst, yu_lst)
			self.log_dict(scores_u, **self.log_params)

		return loss

	def mixmatch(
		self,
		xs_weak: Tensor,
		xu_weak_lst: Tensor,
		ys: Tensor,
		yu_lst: Tensor
	) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
		"""
			Apply MixUp between labeled and unlabeled data.
			Note: xs_weak and xu_weak_lst must have the same number of dimension but they can have a different bsize.

			:param xs_weak: (bsize_s, *features...)
			:param xu_weak_lst: (n_augms * bsize_u, *features...)
			:param ys: (bsize_s, n_classes)
			:param yu_lst: (bsize_u, n_classes)
			:return: The tuple of labeled and unlabeled data mixed : (xs_mixed, xu_mixed, ys_mixed, yu_mixed).
		"""
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
		assert len(xu_weak_lst) > 0
		pred_xu_weak_lst = self.activation(self.model(xu_weak_lst[0]))
		for xu_weak in xu_weak_lst[1:]:
			pred_xu_weak_lst += self.activation(self.model(xu_weak))
		pred_xu_weak_lst /= self.n_augms
		yu = self.sharpen(pred_xu_weak_lst)
		return yu

	def sharpen(self, pred: Tensor) -> Tensor:
		pred = pred ** (1.0 / self.temperature)
		pred = pred / pred.norm(p=1, dim=-1, keepdim=True)
		return pred

	def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_xs = self(xs)
		self.log_dict(self.metric_dict_val(pred_xs, ys), on_epoch=True, on_step=False)

	def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_xs = self(xs)
		self.log_dict(self.metric_dict_test(pred_xs, ys), on_epoch=True, on_step=False)

	def forward(self, x: Tensor) -> Tensor:
		return self.activation(self.model(x))

	def configure_optimizers(self) -> Optimizer:
		return self.optimizer
