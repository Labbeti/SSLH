#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional

import torch

from torch import Tensor

from sslh.metrics.base import Metric


class Recall(Metric):
    def __init__(
        self,
        threshold_input: Optional[float] = 0.5,
        threshold_target: Optional[float] = 0.5,
        dim: Optional[int] = -1,
        reduce_fn: Optional[Callable] = torch.mean,
    ):
        """
        Compute Recall score between binary vectors.
        The recall is intuitively the ability of the classifier to find all the positive samples.

        >>> 'Recall = TP / (TP + FN), where TP = True Positives, FN = False Negatives.'

        :param threshold_input: The threshold value for binarize input vectors. (default: 0.5)
        :param threshold_target: The threshold value for binarize target vectors. (default: 0.5)
        :param dim: The dimension to compute the score. (default: -1)
        :param reduce_fn: The reduction function to apply. (default: torch.mean)
        """
        super().__init__()
        self.dim = dim if dim is not None else ()
        self.threshold_input = threshold_input
        self.threshold_target = threshold_target
        self.reduce_fn = reduce_fn

    def compute_score(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute score with one-hot or multi-hot inputs and targets.

        :param pred: Shape (nb classes) or (nb samplers, nb classes) tensor.
        :param target: Shape (nb classes) or (nb samplers, nb classes) tensor.
        :return: Recall score(s) as tensor in range [0, 1].
        """
        if pred.shape != target.shape:
            raise ValueError(
                f'Invalid input and target shapes for metric "{self.__class__.__name__}". ({pred.shape} != {target.shape})'
            )

        if self.threshold_input is not None:
            pred = pred.ge(self.threshold_input).float()

        if self.threshold_target is not None:
            target = target.ge(self.threshold_target).float()

        assert (
            pred.eq(0.0).logical_or(pred.eq(1.0)).all()
        ), "Input must be binary tensor containing only 0 and 1."
        assert (
            target.eq(0.0).logical_or(target.eq(1.0)).all()
        ), "Target must be binary tensor containing only 0 and 1."

        true_positives = (pred * target).sum(dim=self.dim)
        # Note: TP + FN = Possible positives = sum(target)
        score = true_positives / target.sum(dim=self.dim)
        score[score.isnan()] = 0.0

        if self.reduce_fn is not None:
            score = self.reduce_fn(score)

        return score
