
from pytorch_lightning import LightningModule
from torch import Tensor
from typing import Tuple

from mlu.nn import ForwardDictAffix


class ValModule(LightningModule):
	def __init__(self, pl_module: LightningModule, metric_dict: ForwardDictAffix):
		super().__init__()
		self.pl_module = pl_module
		self.metric_dict = metric_dict

	def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_xs = self(xs)
		scores = self.metric_dict(pred_xs, ys)
		self.log_dict(scores, on_epoch=True, on_step=False)

	def forward(self, *args, **kwargs):
		return self.pl_module(*args, **kwargs)
