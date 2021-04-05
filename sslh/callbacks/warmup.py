
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from typing import Any, Optional


class WarmUpCallback(Callback):
	def __init__(
		self,
		target_value: float,
		start_value: float = 0.0,
		target_num_steps: Optional[int] = None,
		target_ratio_num_steps: Optional[float] = None,
		target_obj: Optional[object] = None,
		target_attribute: Optional[str] = None,
		on_epoch: bool = False,
	):
		if target_num_steps is not None and target_ratio_num_steps is not None:
			raise ValueError("Options 'target_num_steps' and 'target_ratio_num_steps' are mutually exclusive.")

		super().__init__()
		self.target_value = target_value
		self.start_value = start_value
		self.target_num_steps = target_num_steps
		self.target_ratio_num_steps = target_ratio_num_steps
		self.target_obj = target_obj
		self.target_attribute = target_attribute
		self.on_epoch = on_epoch

		self._step = 0
		self._num_steps = 1

		self._update_target()

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
			if self.target_num_steps is not None:
				num_steps = self.target_num_steps
			elif self.target_ratio_num_steps is not None:
				num_steps = int(self.target_ratio_num_steps * trainer.num_training_batches * trainer.max_epochs)
			else:
				num_steps = trainer.num_training_batches * trainer.max_epochs
			self.step_update(step, num_steps)

	def on_train_epoch_end(
		self,
		trainer: Trainer,
		pl_module: LightningModule,
		outputs: Any,
	) -> None:
		if self.on_epoch:
			step = pl_module.current_epoch
			if self.target_num_steps is not None:
				num_steps = self.target_num_steps
			elif self.target_ratio_num_steps is not None:
				num_steps = int(self.target_ratio_num_steps * trainer.max_epochs)
			else:
				num_steps = trainer.max_epochs
			self.step_update(step, num_steps)

	def step_update(self, step: int, num_steps: int):
		self._step = step
		self._num_steps = num_steps
		self._update_target()

	def get_ratio(self) -> float:
		return min(self._step / self._num_steps, 1.0)

	def get_value(self) -> float:
		return (self.target_value - self.start_value) * self.get_ratio() + self.start_value

	def get_step(self) -> int:
		return self._step

	def get_num_steps(self) -> int:
		return self._num_steps

	def has_valid_target(self) -> bool:
		return self.target_obj is not None and self.target_attribute is not None

	def _update_target(self):
		if self.has_valid_target():
			value = self.get_value()
			self.target_obj.__setattr__(self.target_attribute, value)
