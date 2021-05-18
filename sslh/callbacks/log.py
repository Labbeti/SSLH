import gc

from abc import ABC
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from typing import Any


class LogCallback(Callback, ABC):
	def __init__(self, log_on_epoch: bool = True):
		super().__init__()
		self.log_on_epoch = log_on_epoch

	def _log_impl(self, pl_module: LightningModule, **kwargs):
		raise NotImplemented("Abstract method")

	def on_train_epoch_end(
		self,
		trainer: Trainer,
		pl_module: LightningModule,
		outputs: Any,
	) -> None:
		if self.log_on_epoch:
			self._log_impl(pl_module, on_epoch=True, on_step=False)

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
			self._log_impl(pl_module, on_epoch=False, on_step=True)


class LogTensorMemoryCallback(LogCallback):
	def __init__(self, prefix: str = 'train/', log_on_epoch: bool = True):
		super().__init__(log_on_epoch)
		self.prefix = prefix

	def _log_impl(self, pl_module: LightningModule, **kwargs):
		allocated = sum(
			obj.element_size() * obj.nelement()
			for obj in gc.get_objects()
			if isinstance(obj, Tensor)
		)
		allocated = allocated / 2 ** 20
		pl_module.log(f'{self.prefix}memory', allocated, prog_bar=True, **kwargs)


class LogLRCallback(LogCallback):
	def __init__(self, prefix: str = 'train/', log_on_epoch: bool = True):
		super().__init__(log_on_epoch)
		self.prefix = prefix

	def _log_impl(self, pl_module: LightningModule, **kwargs):
		optimizer = pl_module.optimizers()
		learning_rates = [param_group['lr'] for param_group in optimizer.param_groups]

		if len(learning_rates) == 1:
			pl_module.log(f'{self.prefix}lr', learning_rates[0], **kwargs)
		else:
			for i, lr in enumerate(learning_rates):
				pl_module.log(f'{self.prefix}lr_{i}', lr, **kwargs)


class LogAttributeCallback(LogCallback):
	def __init__(self, attr_name: str, prefix: str = 'train/', log_on_epoch: bool = True):
		super().__init__(log_on_epoch)
		self.attr_name = attr_name
		self.prefix = prefix

	def _log_impl(self, pl_module: LightningModule, **kwargs):
		attr_value = pl_module.__getattribute__(self.attr_name)
		pl_module.log(f'{self.prefix}{self.attr_name}', attr_value, **kwargs)


class LogHParamsCallback(LogCallback):
	def __init__(self, prefix: str = 'train/', log_on_epoch: bool = True):
		super().__init__(log_on_epoch)
		self.prefix = prefix
		self.log_on_epoch = log_on_epoch

	def _log_impl(self, pl_module: LightningModule, **kwargs):
		hparam = pl_module.hparams
		for hparam_name, hparam_value in hparam.items():
			pl_module.log(f'{self.prefix}{hparam_name}', hparam_value, **kwargs)
