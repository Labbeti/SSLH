
import logging
import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from typing import Dict, List, Optional, Tuple

from mlu.nn import ForwardDictAffix


class TestModule(LightningModule):
	def __init__(self, module: Module, metric_dict: Optional[Dict[str, Module]], prefix: str) -> None:
		"""
			LightningModule wrapper for module and a metric dict.

			Example :

			>>> from pytorch_lightning import Trainer
			>>> from mlu.metrics.classification.categorical_accuracy import CategoricalAccuracy
			>>> model = ...
			>>> trainer = Trainer(...)
			>>> test_dataloader = ...
			>>> test_module = TestModule(model, ForwardDictAffix(acc=CategoricalAccuracy()))
			>>> trainer.test(test_module, test_dataloader)

			:param module: The module to wrap for testing the forward output.
			:param metric_dict: The metric dict object.
			:param prefix: The prefix used in metrics names.
		"""
		super().__init__()
		self.module = module
		self.metric_dict = ForwardDictAffix(metric_dict, prefix=prefix)

	def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
		xs, ys = batch
		pred_xs = self(xs)
		scores = self.metric_dict(pred_xs, ys)
		self.log_dict(scores, on_epoch=True, on_step=False, logger=False, prog_bar=False)
		return scores

	def test_epoch_end(self, scores_lst: List[Dict[str, Tensor]]) -> None:
		scores = {
			name: torch.stack([scores[name] for scores in scores_lst]).mean().item()
			for name in scores_lst[0].keys()
		}
		self.logger.log_hyperparams({}, scores)

	def forward(self, *args, **kwargs):
		return self.module(*args, **kwargs)
