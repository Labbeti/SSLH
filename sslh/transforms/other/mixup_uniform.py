#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

import torch

from torch import Tensor
from torch.nn import Module


class MixUpUniform(Module):
    def __init__(self):
        super().__init__()
        self._lambda = torch.zeros(1)

    def forward(
        self,
        batch_a: Tensor,
        batch_b: Tensor,
        labels_a: Tensor,
        labels_b: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply MixUp to batches and labels.
        """
        assert batch_a.shape == batch_b.shape and labels_a.shape == labels_b.shape

        mask = torch.rand_like(batch_a)
        self._lambda = mask.sum() / torch.prod(torch.as_tensor(mask.shape))

        batch_mix = batch_a * mask + batch_b * (1.0 - mask)
        labels_mix = labels_a * self._lambda + labels_b * (1.0 - self._lambda)
        return batch_mix, labels_mix

    def get_last_lambda(self) -> float:
        return self._lambda.item()
