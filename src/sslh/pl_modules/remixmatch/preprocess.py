#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Tuple

from torch import nn


class ReMixMatchUnlabeledPreProcess(nn.Module):
    """
    Compose transform_weak and transform_strong for unlabeled data.

    Note: (weak(data), (strong(data), strong(data), ...))
    """

    def __init__(
        self,
        transform_weak: Callable,
        transform_strong: Callable,
        n_strong_augs: int,
    ) -> None:
        super().__init__()
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        self.n_strong_augs = n_strong_augs

    def forward(self, data: Any) -> Tuple[Any, Tuple[Any, ...]]:
        weak_data = self.transform_weak(data)
        strong_data_list = tuple(
            self.transform_strong(data) for _ in range(self.n_strong_augs)
        )
        return weak_data, strong_data_list
