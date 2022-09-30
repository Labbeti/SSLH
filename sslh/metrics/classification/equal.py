#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional

import torch

from torch import Tensor

from sslh.metrics.base import Metric


class Equal(Metric):
    def __init__(
        self, dim: Optional[int] = -1, reduce_fn: Callable = torch.mean
    ) -> None:
        """
        Equal metric along a dimension.

        :param dim: The dimension to compute the score.
        """
        super().__init__()
        self.dim = dim
        self.reduce_fn = reduce_fn

    def compute_score(self, pred: Tensor, target: Tensor) -> Tensor:
        score = pred.eq(target).all(self.dim).float()
        score = self.reduce_fn(score)
        return score
