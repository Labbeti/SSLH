
import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from typing import List, Tuple

from mlu.metrics import MetricDict


class TestStackModule(LightningModule):
	def __init__(self, experiment_module: LightningModule, metric_dict: MetricDict):
		super().__init__()
		self.experiment_module = experiment_module
		self.metric_dict = metric_dict

	def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tuple[Tensor, Tensor]:
		xs, ys = batch
		pred_xs = self.experiment_module(xs)
		return pred_xs, ys

	def test_epoch_end(self, outputs: List[Tuple[Tensor, Tensor]]):
		list_pred = torch.vstack([pred for pred, _ in outputs])
		list_ys = torch.vstack([ys for _, ys in outputs])
		scores = self.metric_dict(list_pred, list_ys)
		self.log_dict(scores, on_epoch=True, on_step=False)

	def forward(self, *args, **kwargs):
		return self.experiment_module(*args, **kwargs)
