
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from typing import Any, Protocol


class Steppable(Protocol):
	def step(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, **kwargs):
		raise NotImplementedError('Protocol abstract method')


class CallbackWrapper(Callback):
	def __init__(
		self,
		steppable: Steppable,
		on_train_epoch_end_: bool = False,
		on_train_batch_end_: bool = True,
	):
		super().__init__()
		self.steppable = steppable
		self.on_train_epoch_end_ = on_train_epoch_end_
		self.on_train_batch_end_ = on_train_batch_end_

	def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any) -> None:
		if self.on_train_epoch_end_:
			self.steppable.step(trainer, pl_module, outputs)

	def on_train_batch_end(
		self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
	) -> None:
		if self.on_train_batch_end_:
			self.steppable.step(trainer, pl_module, outputs, batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
