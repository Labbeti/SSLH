#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable

import torch

from torch import Tensor

from sslh.metrics.classification.mse import MSE


class RMSE(MSE):
    def __init__(self, reduce_fn: Callable = torch.mean):
        super().__init__(reduce_fn=reduce_fn)

    def compute_score(self, pred: Tensor, target: Tensor) -> Tensor:
        mse = super().compute_score(pred, target)
        return mse.sqrt()
