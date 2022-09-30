#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable

import torch

from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from torch import Tensor

from sslh.metrics.base import Metric


class DPrime(Metric):
    def __init__(self, average: str = "macro", reduce_fn: Callable = torch.mean):
        """
        DPrime metric.

        Note: If score == 0 : bad score, low difference between 'noise' and inputs.

        Backend: sklearn and scipy.

        :param average: The type of D' score to compute. (default: 'macro')
        :param reduce_fn: The reduction function to apply. (default: torch.mean)
        """
        super().__init__()
        self.average = average
        self.reduce_fn = reduce_fn

    def compute_score(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute DPrime score on prediction and target.

        :param pred: (n_classes) or (batch_size, n_classes) tensor
        :param target: (n_classes) or (batch_size, n_classes) tensor
        :return: The DPrime score as scalar tensor.
        """
        assert pred.shape == target.shape
        assert len(pred.shape) == 2

        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        roc_auc = roc_auc_score(y_true=target, y_score=pred, average=self.average)
        score = (2 ** 0.5) * norm.ppf(roc_auc)
        score = torch.as_tensor(score)
        score = self.reduce_fn(score)

        return score


def d_prime_from_auc(auc: float) -> float:
    result = norm.ppf(auc) * 2 ** 0.5
    return result
