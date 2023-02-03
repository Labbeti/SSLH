#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import Tensor

from sslh.metrics.base import Metric


class EqMetric(Metric):
    def __init__(self, dim: int):
        """
        Equal metric along a dimension.

        :param dim: The dimension to compute the score.
        """
        super().__init__()
        self.dim = dim

    def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
        return input_.eq(target).all(self.dim).float()
