#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional

import torch

from torch import Tensor

from sslh.metrics.classification.precision import Precision
from sslh.metrics.classification.recall import Recall
from sslh.metrics.base import Metric


class FScore(Metric):
    def __init__(
        self,
        threshold_input: Optional[float] = 0.5,
        threshold_target: Optional[float] = 0.5,
        beta: float = 1.0,
        dim: Optional[int] = -1,
        reduce_fn: Optional[Callable] = torch.mean,
    ):
        """
        FScore metric. (micro).

        >>> 'FScore = 2 * precision * recall / (recall + precision)'

        :param threshold_input: The threshold value for binarize input vectors. (default: 0.5)
        :param threshold_target: The threshold value for binarize target vectors. (default: 0.5)
        :param beta: The beta fscore parameter. (default: 1.0)
        :param dim: The dimension to compute the score. (default: -1)
        :param reduce_fn: The reduction function to apply. (default: torch.mean)
        """
        super().__init__()
        self.beta = beta
        self.threshold_input = threshold_input
        self.threshold_target = threshold_target
        self.reduce_fn = reduce_fn

        self.recall = Recall(
            threshold_input=None,
            threshold_target=None,
            dim=dim,
            reduce_fn=None,
        )
        self.precision = Precision(
            threshold_input=None,
            threshold_target=None,
            dim=dim,
            reduce_fn=None,
        )

    def compute_score(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute score with one-hot or multi-hot inputs and targets.

        :param pred: Shape (nb classes) or (nb samplers, nb classes) binary tensor.
        :param target: Shape (nb classes) or (nb samplers, nb classes) binary tensor.
        :return: FScore score(s) as tensor in range [0, 1].
        """
        if self.threshold_input is not None:
            pred = pred.ge(self.threshold_input).float()

        if self.threshold_target is not None:
            target = target.ge(self.threshold_target).float()

        recall = self.recall(pred, target)
        precision = self.precision(pred, target)

        score = (
            (1.0 + self.beta ** 2)
            * precision
            * recall
            / (self.beta ** 2 * precision + recall)
        )
        score[score.isnan()] = 0.0

        if self.reduce_fn is not None:
            score = self.reduce_fn(score)

        return score
