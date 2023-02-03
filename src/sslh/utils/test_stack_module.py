#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Tuple

import torch

from pytorch_lightning import LightningModule
from torch import nn, Tensor

from sslh.nn.utils import ForwardDictAffix
from sslh.utils.custom_logger import CustomTensorboardLogger


class TestStackModule(LightningModule):
    def __init__(
        self,
        module: nn.Module,
        metric_dict: Optional[Dict[str, nn.Module]],
        prefix: str,
    ) -> None:
        """
        :param module: TODO
        :param metric_dict: TODO
        :param prefix: TODO
        """
        super().__init__()
        self.module = module
        self.metric_dict = ForwardDictAffix(metric_dict, prefix=prefix)

    def test_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Tuple[Tensor, Tensor]:
        xs, ys = batch
        probs_xs = self(xs)
        return probs_xs, ys

    def test_epoch_end(self, outputs: List[Tuple[Tensor, Tensor]]) -> None:
        list_pred = torch.vstack([pred for pred, _ in outputs])
        list_ys = torch.vstack([ys for _, ys in outputs])
        scores = self.metric_dict(list_pred, list_ys)
        self.log_dict(scores, on_epoch=True, on_step=False, logger=False, prog_bar=True)

        if isinstance(self.logger, CustomTensorboardLogger):
            self.logger.log_hyperparams({}, scores)
        else:
            self.logger.log_metrics(scores, None)  # type: ignore

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
