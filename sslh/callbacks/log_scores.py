
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch.nn import Module
from typing import Any, Dict


class LogScores(Callback):
	def __init__(
			self,
			activation: Module,
			metric_dict_train: Dict[str, Module],
			metric_dict_val: Dict[str, Module],
			metric_dict_test: Dict[str, Module],
			log_on_epoch: bool,
	):
		super().__init__()
		self.activation = activation
		self.metrics_train = metric_dict_train
		self.metrics_val = metric_dict_val
		self.metrics_test = metric_dict_test
		self.log_on_epoch = log_on_epoch

		self.prefix_train = 'train/'
		self.prefix_val = 'val/'
		self.prefix_test = 'test/'

	def on_train_batch_end(
			self,
			trainer: Trainer,
			pl_module: LightningModule,
			outputs: Dict[str, Any],
			batch: Any,
			batch_idx: int,
			dataloader_idx: int,
	) -> None:
		self.log_scores(pl_module, outputs, self.metrics_train, self.prefix_train)

	def on_validation_batch_end(
			self,
			trainer: Trainer,
			pl_module: LightningModule,
			outputs: Dict[str, Any],
			batch: Any,
			batch_idx: int,
			dataloader_idx: int,
	) -> None:
		self.log_scores(pl_module, outputs, self.metrics_val, self.prefix_val)

	def on_test_batch_end(
			self,
			trainer: Trainer,
			pl_module: LightningModule,
			outputs: Dict[str, Any],
			batch: Any,
			batch_idx: int,
			dataloader_idx: int,
	) -> None:
		self.log_scores(pl_module, outputs, self.metrics_test, self.prefix_test)

	def log_scores(
			self,
			pl_module: LightningModule,
			outputs: Dict[str, Any],
			metrics: Dict[str, Module],
			prefix: str,
	):
		metric_inputs = outputs['metric_inputs']

		for logits, y, suffix in metric_inputs:
			pred = self.activation(logits)
			scores = {
				f'{prefix}{name}{suffix}': metric(pred, y) for name, metric in metrics.items()
			}
			pl_module.log_dict(scores, on_epoch=self.log_on_epoch, on_step=not self.log_on_epoch)

		if 'others' in outputs.keys():
			others = outputs['others']
			others = {f'{prefix}{name}': value for name, value in others.items()}
			pl_module.log_dict(others, on_epoch=self.log_on_epoch, on_step=not self.log_on_epoch)
