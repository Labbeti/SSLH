#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from torch import Tensor
from torch.nn import Module
from typing import Tuple, Union

from sslh.transforms.other.mixup import MixUpModule


class MixUpBatchItSelf(Module):
    """
    Apply MixUp transform with the same batch in a different order. See MixUpModule for details.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        apply_max: bool = False,
        return_shuffled_batches: bool = False,
    ) -> None:
        super().__init__()
        self.mixup = MixUpModule(alpha, apply_max)
        self.return_shuffled_batches = return_shuffled_batches

    def forward(
        self,
        batch: Tensor,
        labels: Tensor,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        assert batch.shape[0] == labels.shape[0]
        batch_size = batch.shape[0]
        indexes = torch.randperm(batch_size)
        batch_shuffle = batch[indexes]
        labels_shuffle = labels[indexes]

        batch_mix, labels_mix = self.mixup(batch, batch_shuffle, labels, labels_shuffle)
        if not self.return_shuffled_batches:
            return batch_mix, labels_mix
        else:
            return batch_mix, labels_mix, batch_shuffle, labels_shuffle

    def get_last_lambda(self) -> float:
        return self.mixup.get_last_lambda()
