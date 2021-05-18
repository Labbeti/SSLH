
import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from typing import Dict, List, Optional, Tuple

from mlu.nn import ForwardDictAffix


class TestStackModule(LightningModule):
	def __init__(self, module: Module, metric_dict: Optional[Dict[str, Module]], prefix: str):
		"""
			:param module: TODO
			:param metric_dict: TODO
			:param prefix: TODO
		"""
		super().__init__()
		self.module = module
		self.metric_dict = ForwardDictAffix(metric_dict, prefix=prefix)

	def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tuple[Tensor, Tensor]:
		xs, ys = batch
		pred_xs = self(xs)
		return pred_xs, ys

	def test_epoch_end(self, outputs: List[Tuple[Tensor, Tensor]]):
		list_pred = torch.vstack([pred for pred, _ in outputs])
		list_ys = torch.vstack([ys for _, ys in outputs])
		scores = self.metric_dict(list_pred, list_ys)
		self.log_dict(scores, on_epoch=True, on_step=False, logger=False, prog_bar=True)
		self.logger.log_hyperparams({}, scores)

	def forward(self, *args, **kwargs):
		return self.module(*args, **kwargs)
