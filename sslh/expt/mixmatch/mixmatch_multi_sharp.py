
from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer
from typing import Dict, Optional

from mlu.nn import CrossEntropyWithVectors
from sslh.expt.mixmatch.mixmatch import MixMatch


class MixMatchMultiSharp(MixMatch):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		activation: Module = Softmax(dim=-1),
		criterion_s: Module = CrossEntropyWithVectors(reduction='mean'),
		criterion_u: Module = CrossEntropyWithVectors(reduction='mean'),
		lambda_u: float = 1.0,
		n_augms: int = 2,
		alpha: float = 0.75,
		sharpen_threshold: float = 0.75,
		train_metrics: Optional[Dict[str, Module]] = None,
		val_metrics: Optional[Dict[str, Module]] = None,
		log_on_epoch: bool = True,
	):
		"""
			MixMatch with an experimental multi-hot sharpening (MMM) LightningModule.

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
			:param sharpen_threshold: Sharpen multihot threshold param. (default: 0.75)
			:param alpha: The mixup alpha parameter. A higher value means a stronger mix between labeled and unlabeled data.
				(default: 0.75)
			:param train_metrics: An optional dictionary of metrics modules for training.
				(default: None)
			:param val_metrics: An optional dictionary of metrics modules for validation.
				(default: None)
			:param log_on_epoch: If True, log only the epoch means of each train metric score.
				(default: True)
		"""
		super().__init__(
			model=model,
			optimizer=optimizer,
			activation=activation,
			criterion_s=criterion_s,
			criterion_u=criterion_u,
			lambda_u=lambda_u,
			n_augms=n_augms,
			temperature=0.0,
			alpha=alpha,
			train_metrics=train_metrics,
			val_metrics=val_metrics,
			log_on_epoch=log_on_epoch,
		)
		self.sharpen_threshold = sharpen_threshold
		self.save_hyperparameters({'sharpen_threshold': sharpen_threshold})

	def sharpen(self, pred: Tensor) -> Tensor:
		probs = pred + 1.0 - self.sharpen_threshold
		pred = pred * probs
		return pred
