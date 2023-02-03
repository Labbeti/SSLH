#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable

import torch

from sklearn.metrics import roc_auc_score
from torch import Tensor

from sslh.metrics.base import Metric


class RocAuc(Metric):
    def __init__(self, average: str = "macro", reduce_fn: Callable = torch.mean):
        """
        Compute mean Area Under the Receiver Operating Characteristic Curve (ROC AUC) score.
        Backend: scikit-learn

        :param average: The type of ROC AUC score to compute. (default: 'macro')
        :param reduce_fn: The reduction function to apply. (default: torch.mean)
        """
        super().__init__()
        self.average = average
        self.reduce_fn = reduce_fn

    def compute_score(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute ROC AUC score on prediction and target.

        :param pred: (batch_size, n_classes) tensor
        :param target: (batch_size, n_classes) tensor
        :return: The ROC AUC score as scalar tensor.
        """
        assert pred.shape == target.shape
        assert len(pred.shape) == 2

        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        score = roc_auc_score(y_true=target, y_score=pred, average=self.average)
        score = torch.as_tensor(score)
        score = self.reduce_fn(score)

        return score
