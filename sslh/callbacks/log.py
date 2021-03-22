
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from typing import Any


class LogLRCallback(Callback):
	def __init__(self, optimizer_attr_name: str = "optimizer", prefix: str = "train/", log_on_epoch: bool = True):
		super().__init__()
		self.optimizer_attr_name = optimizer_attr_name
		self.prefix = prefix
		self.log_on_epoch = log_on_epoch

	def log_lr(self, pl_module: LightningModule, **kwargs):
		optimizer = pl_module.__getattribute__(self.optimizer_attr_name)
		learning_rates = [param_group["lr"] for param_group in optimizer.param_groups]

		if len(learning_rates) == 1:
			pl_module.log(f"{self.prefix}lr", learning_rates[0], **kwargs)
		else:
			for i, lr in enumerate(learning_rates):
				pl_module.log(f"{self.prefix}lr_{i}", lr, **kwargs)

	def on_train_epoch_end(
		self,
		trainer: Trainer,
		pl_module: LightningModule,
		outputs: Any,
	) -> None:
		if self.log_on_epoch:
			self.log_lr(pl_module, on_epoch=True, on_step=False)

	def on_train_batch_end(
		self,
		trainer: Trainer,
		pl_module: LightningModule,
		outputs: Any,
		batch: Any,
		batch_idx: int,
		dataloader_idx: int,
	) -> None:
		if not self.log_on_epoch:
			self.log_lr(pl_module, on_epoch=False, on_step=True)


class LogAttributeCallback(Callback):
	def __init__(self, attr_name: str, prefix: str = "train/", log_on_epoch: bool = True):
		self.attr_name = attr_name
		self.prefix = prefix
		self.log_on_epoch = log_on_epoch

	def log_attribute(self, pl_module: LightningModule, **kwargs):
		attr_value = pl_module.__getattribute__(self.attr_name)
		pl_module.log(f"{self.prefix}{self.attr_name}", attr_value, **kwargs)

	def on_train_epoch_end(
		self,
		trainer: Trainer,
		pl_module: LightningModule,
		outputs: Any,
	) -> None:
		if self.log_on_epoch:
			self.log_attribute(pl_module, on_epoch=True, on_step=False)

	def on_train_batch_end(
		self,
		trainer: Trainer,
		pl_module: LightningModule,
		outputs: Any,
		batch: Any,
		batch_idx: int,
		dataloader_idx: int,
	) -> None:
		if not self.log_on_epoch:
			self.log_attribute(pl_module, on_epoch=False, on_step=True)


class LogHParamsCallback(Callback):
	def __init__(self, prefix: str = "train/", log_on_epoch: bool = True):
		self.prefix = prefix
		self.log_on_epoch = log_on_epoch

	def log_hparams(self, pl_module: LightningModule, **kwargs):
		hparam = pl_module.hparams
		for hparam_name, hparam_value in hparam.items():
			pl_module.log(f"{self.prefix}{hparam_name}", hparam_value, **kwargs)

	def on_train_epoch_end(
		self,
		trainer: Trainer,
		pl_module: LightningModule,
		outputs: Any,
	) -> None:
		if self.log_on_epoch:
			self.log_hparams(pl_module, on_epoch=True, on_step=False)

	def on_train_batch_end(
		self,
		trainer: Trainer,
		pl_module: LightningModule,
		outputs: Any,
		batch: Any,
		batch_idx: int,
		dataloader_idx: int,
	) -> None:
		if not self.log_on_epoch:
			self.log_hparams(pl_module, on_epoch=False, on_step=True)
