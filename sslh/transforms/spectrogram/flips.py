#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Union

import torch

from torch import Tensor

from sslh.transforms.base import SpectrogramTransform


class Flip(SpectrogramTransform):
    def __init__(self, dim: Union[int, Tuple[int, ...]], p: float = 1.0) -> None:
        super().__init__(p=p)
        self.dim = dim

    def process(self, data: Tensor) -> Tensor:
        if isinstance(self.dim, int):
            dims = (self.dim,)
        else:
            dims = tuple(self.dim)
        return torch.flip(data, dims)


class HorizontalFlip(Flip):
    """Flip horizontally. (from left to right, axis is vertical)"""

    def __init__(self, p: float = 1.0) -> None:
        super().__init__(dim=-1, p=p)


class VerticalFlip(Flip):
    """Flip vertically. (from top to bottom, axis is horizontal)"""

    def __init__(self, p: float = 1.0) -> None:
        super().__init__(dim=-2, p=p)
