
from pytorch_lightning import LightningModule
from torch import Tensor
from typing import Tuple

from mlu.metrics import MetricDict


class TestModule(LightningModule):
	def __init__(self, experiment_module: LightningModule, metric_dict: MetricDict):
		super().__init__()
		self.experiment_module = experiment_module
		self.metric_dict = metric_dict

	def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_xs = self.experiment_module(xs)
		scores = self.metric_dict(pred_xs, ys)
		self.log_dict(scores, on_epoch=True, on_step=False)

	def forward(self, *args, **kwargs):
		return self.experiment_module(*args, **kwargs)
