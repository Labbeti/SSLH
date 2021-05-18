
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from typing import Any, Optional


class WarmUp:
	def __init__(
		self,
		target_value: float,
		n_steps: Optional[int] = None,
		target_obj: Optional[object] = None,
		target_attribute: Optional[str] = None,
		start_value: float = 0.0,
	):
		super().__init__()
		self.target_value = target_value
		self.target_obj = target_obj
		self.target_attribute = target_attribute
		self.start_value = start_value

		self._step = 0
		self._n_steps = n_steps if n_steps is not None else 1

		self._update_target()

	def step(self):
		self._step += 1
		self._update_target()

	def set_and_update(self, step: int, n_steps: int):
		self._step = step
		self._n_steps = n_steps
		self._update_target()

	def get_ratio(self) -> float:
		return min(self._step / self._n_steps, 1.0)

	def get_value(self) -> float:
		return (self.target_value - self.start_value) * self.get_ratio() + self.start_value

	def get_step(self) -> int:
		return self._step

	def get_n_steps(self) -> int:
		return self._n_steps

	def has_valid_target(self) -> bool:
		return self.target_obj is not None and self.target_attribute is not None

	def _update_target(self):
		if self.has_valid_target():
			value = self.get_value()
			self.target_obj.__setattr__(self.target_attribute, value)


class WarmUpCallback(WarmUp, Callback):
	def __init__(
		self,
		target_value: float,
		start_value: float = 0.0,
		n_steps: Optional[int] = None,
		ratio_n_steps: Optional[float] = None,
		target_obj: Optional[object] = None,
		target_attribute: Optional[str] = None,
		on_epoch: bool = False,
	):
		if n_steps is not None and ratio_n_steps is not None:
			raise ValueError('Options "target_n_steps" and "target_ratio_n_steps" are mutually exclusive.')

		WarmUp.__init__(
			self,
			target_value=target_value,
			start_value=start_value,
			n_steps=n_steps,
			target_obj=target_obj,
			target_attribute=target_attribute,
		)
		Callback.__init__(self)

		self.target_ratio_n_steps = ratio_n_steps
		self.on_epoch = on_epoch

	def on_train_batch_end(
		self,
		trainer: Trainer,
		pl_module: LightningModule,
		outputs: Any,
		batch: Any,
		batch_idx: int,
		dataloader_idx: int
	) -> None:
		if not self.on_epoch:
			step = pl_module.global_step
			if self.target_ratio_n_steps is not None:
				n_steps = int(self.target_ratio_n_steps * trainer.num_training_batches * trainer.max_epochs)
			elif self.get_n_steps() is not None:
				n_steps = self.get_n_steps()
			else:
				n_steps = trainer.num_training_batches * trainer.max_epochs
			self.set_and_update(step, n_steps)

	def on_train_epoch_end(
		self,
		trainer: Trainer,
		pl_module: LightningModule,
		outputs: Any,
	) -> None:
		if self.on_epoch:
			step = pl_module.current_epoch
			if self.target_ratio_n_steps is not None:
				n_steps = int(self.target_ratio_n_steps * trainer.max_epochs)
			elif self.get_n_steps() is not None:
				n_steps = self.get_n_steps()
			else:
				n_steps = trainer.max_epochs
			self.set_and_update(step, n_steps)
