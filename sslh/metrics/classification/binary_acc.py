#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional

import torch

from torch import Tensor

from sslh.metrics.base import Metric


class BinaryAccuracy(Metric):
    def __init__(
        self,
        threshold_input: Optional[float] = 0.5,
        threshold_target: Optional[float] = None,
        reduce_fn: Callable = torch.mean,
    ):
        """
        Binary Accuracy metric.
        Compute the accuracy between two multihot vectors.

        :param threshold_input: The optional threshold to apply to inputs.
        :param threshold_target: The optional threshold to apply to targets.
        :param reduce_fn: The reduction function to apply to score.
        """
        super().__init__()
        self.threshold_input = threshold_input
        self.threshold_target = threshold_target
        self.reduce_fn = reduce_fn

    def compute_score(self, input_: Tensor, target: Tensor) -> Tensor:
        assert input_.shape == target.shape
        assert 0 <= len(input_.shape) <= 2

        if self.threshold_input is not None:
            input_ = input_.ge(self.threshold_input).float()

        if self.threshold_target is not None:
            target = target.ge(self.threshold_target).float()

        score = input_.eq(target).float()
        score = self.reduce_fn(score)
        return score
