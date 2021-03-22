
import math

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Any, Callable


class LRSchedulerCallback(Callback, LambdaLR):
	def __init__(
		self,
		optimizer: Optimizer,
		lr_lambda: Callable[[int, int], float],
		on_train_epoch_end: bool = True,
		on_train_batch_end: bool = False,
	):
		# Note: self.lr_lambda and self.num_epochs must be defined before super().__init__ call !
		self.lr_lambda = lr_lambda
		self.num_epochs = 1
		super().__init__(optimizer=optimizer, lr_lambda=self._lr_lambda_torch)
		self.on_train_epoch_end_ = on_train_epoch_end
		self.on_train_batch_end_ = on_train_batch_end

	def on_train_epoch_end(
		self,
		trainer: Trainer,
		pl_module: LightningModule,
		outputs: Any,
	) -> None:
		if self.on_train_epoch_end_:
			epoch = pl_module.current_epoch
			self.num_epochs = trainer.max_epochs
			self.step(epoch)

	def on_train_batch_end(
		self,
		trainer: Trainer,
		pl_module: LightningModule,
		outputs: Any,
		batch: Any,
		batch_idx: int,
		dataloader_idx: int,
	) -> None:
		if self.on_train_batch_end_:
			step = pl_module.global_step
			self.num_epochs = trainer.num_training_batches * trainer.max_epochs
			self.step(step)

	def _lr_lambda_torch(self, step: int) -> float:
		return self.lr_lambda(step, self.num_epochs)


class CosineScheduler(LRSchedulerCallback):
	def __init__(
		self,
		optimizer: Optimizer,
		on_epoch: bool = True,
	):
		super().__init__(
			optimizer=optimizer,
			lr_lambda=lambda step, num_steps: math.cos(7.0 / 16.0 * math.pi * min(step / num_steps, 1.0)),
			on_train_epoch_end=on_epoch,
			on_train_batch_end=not on_epoch,
		)


class SoftCosineScheduler(LRSchedulerCallback):
	def __init__(
		self,
		optimizer: Optimizer,
		on_epoch: bool = True,
	):
		super().__init__(
			optimizer=optimizer,
			lr_lambda=lambda step, num_steps: 0.5 * (1.0 + math.cos((step - 1) * math.pi / num_steps)),
			on_train_epoch_end=on_epoch,
			on_train_batch_end=not on_epoch,
		)
