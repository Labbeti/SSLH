#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable

import torch

from sklearn.metrics import recall_score
from torch import Tensor

from sslh.metrics.base import Metric


class UAR(Metric):
    """
    Compute the Unweighted Average Recall (UAR) score.
    """

    def __init__(
        self,
        dim: int = -1,
        vector_input: bool = True,
        vector_target: bool = True,
        average: str = "macro",
        reduce_fn: Callable = torch.mean,
    ):
        super().__init__()
        self.dim = dim
        self.vector_input = vector_input
        self.vector_target = vector_target
        self.reduce_fn = reduce_fn
        self.average = average

    def compute_score(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.vector_input:
            pred = pred.argmax(dim=self.dim)

        if self.vector_target:
            target = target.argmax(dim=self.dim)

        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        score = recall_score(y_true=target, y_pred=pred, average=self.average)
        score = torch.as_tensor(score)
        score = self.reduce_fn(score)

        return score
