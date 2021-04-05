
from pytorch_lightning import Callback, LightningModule, Trainer
from typing import Any

from mlu.metrics import MetricDict


class ValidationCallback(Callback):
	def __init__(self, metric_dict_val: MetricDict):
		super().__init__()
		self.metric_dict_val = metric_dict_val

	def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any):
		self.validate(trainer, pl_module)

	def validate(self, trainer: Trainer, pl_module: LightningModule):
		val_dataloaders = trainer.val_dataloaders

		for dataloader in val_dataloaders:
			for batch in dataloader:
				xs, ys = batch
				pred_xs = pl_module(xs)
				pl_module.log_dict(self.metric_dict_val(pred_xs, ys), on_epoch=True, on_step=False)
