#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from typing import Callable, Optional

import torch

from torch import Tensor

from sslh.metrics.base import Metric


class BCEMetric(Metric):
    def __init__(
        self,
        clamp_min: float = -math.inf,
        clamp_max: float = math.inf,
        reduce_fn: Optional[Callable] = torch.mean,
    ) -> None:
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.reduce_fn = reduce_fn

    def compute_score(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape != target.shape:
            raise RuntimeError(f"Mismatch shapes {pred.shape} != {target.shape}.")

        pred = torch.clamp(pred, min=self.clamp_min, max=self.clamp_max)

        scores = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)

        if self.reduce_fn is not None:
            scores = self.reduce_fn(scores)
        return scores
