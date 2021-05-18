
import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Dict, Optional, Tuple

from mlu.nn import ForwardDictAffix, CrossEntropyWithVectors, OneHot


class FixMatch(LightningModule):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Module = Softmax(dim=-1),
		criterion_s: Module = CrossEntropyWithVectors(reduction='none'),
		criterion_u: Module = CrossEntropyWithVectors(reduction='none'),
		target_transform: Module = OneHot(n_classes=10),
		lambda_u: float = 1.0,
		threshold: float = 0.95,
		train_metrics: Optional[Dict[str, Module]] = None,
		val_metrics: Optional[Dict[str, Module]] = None,
		log_on_epoch: bool = True,
	):
		"""
			FixMatch (FM) LightningModule.

			:param model: The PyTorch Module to train.
				The forward() must return logits for classify the data.
			:param optimizer: The PyTorch optimizer to use.
			:param activation: The activation function of the model.
				(default: Softmax(dim=-1))
			:param criterion_s: The loss component 'L_s' of RMM.
				(default: CrossEntropyWithVectors())
			:param criterion_u: The loss component 'L_u' of RMM.
				(default: CrossEntropyWithVectors())
			:param target_transform: The target transform for convert binarized labels to vector probabilities.
				(default: OneHot(n_classes=10))
			:param lambda_u: The coefficient of the 'L_u' component.
				(default: 1.0)
			:param threshold: The confidence threshold 'tau' used for the mask of the 'L_u' component.
				(default: 0.95)
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
		self.target_transform = target_transform
		self.criterion_s = criterion_s
		self.criterion_u = criterion_u
		self.threshold = threshold
		self.lambda_u = lambda_u

		self.metric_dict_train_s = ForwardDictAffix(train_metrics, prefix='train/', suffix='_s')
		self.metric_dict_train_u_pseudo = ForwardDictAffix(train_metrics, prefix='train/', suffix='_u')
		self.metric_dict_val = ForwardDictAffix(val_metrics, prefix='val/')
		self.metric_dict_test = ForwardDictAffix(val_metrics, prefix='test/')

		self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)

		self.save_hyperparameters({
			'experiment': self.__class__.__name__,
			'model': model.__class__.__name__,
			'optimizer': optimizer.__class__.__name__,
			'activation': activation.__class__.__name__,
			'criterion_s': criterion_s.__class__.__name__,
			'criterion_u': criterion_u.__class__.__name__,
			'target_transform': target_transform.__class__.__name__,
			'lambda_u': lambda_u,
			'threshold': threshold,
		})

	def training_step(
		self,
		batch: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]],
		batch_idx: int,
	) -> Tensor:
		(xs_weak, ys), (xu_weak, xu_strong) = batch

		# Compute pseudo-labels 'yu' and mask
		yu, mask = self.guess_label_and_mask(xu_weak)

		# Compute predictions on xs and xu
		logits_xs_weak = self.model(xs_weak)
		logits_xu_strong = self.model(xu_strong)

		pred_xs_weak = self.activation(logits_xs_weak)
		pred_xu_strong = self.activation(logits_xu_strong)

		# Criterion (loss_s of shape bsize_s, loss_u of shape bsize_u)
		loss_s = self.criterion_s(pred_xs_weak, ys)
		loss_u = self.criterion_u(pred_xu_strong, yu)

		loss_s = torch.mean(loss_s)
		loss_u = torch.mean(loss_u * mask)

		loss = loss_s + self.lambda_u * loss_u

		with torch.no_grad():
			scores = {'train/loss': loss, 'train/loss_s': loss_s, 'train/loss_u': loss_u, 'train/mask': mask.mean()}
			scores = {k: v.cpu() for k, v in scores.items()}
			self.log_dict(scores, **self.log_params)

			scores_s = self.metric_dict_train_s(self.activation(logits_xs_weak), ys)
			self.log_dict(scores_s, **self.log_params)

			scores_u = self.metric_dict_train_u_pseudo(self.activation(logits_xu_strong), yu)
			self.log_dict(scores_u, **self.log_params)

		return loss

	def guess_label_and_mask(self, xu_weak: Tensor) -> Tuple[Tensor, Tensor]:
		with torch.no_grad():
			pred_xu_weak = self.activation(self.model(xu_weak))
			probabilities_max, indices_max = pred_xu_weak.max(dim=-1)
			mask = probabilities_max.ge(self.threshold).to(pred_xu_weak.dtype)
			yu = self.target_transform(indices_max)
			return yu, mask

	def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_xs = self(xs)
		self.log_dict(self.metric_dict_val(pred_xs, ys), on_epoch=True, on_step=False)

	def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_xs = self(xs)
		self.log_dict(self.metric_dict_test(pred_xs, ys), on_epoch=True, on_step=False)

	def forward(self, x: Tensor) -> Tensor:
		pred_x = self.activation(self.model(x))
		return pred_x

	def configure_optimizers(self) -> Optimizer:
		return self.optimizer
