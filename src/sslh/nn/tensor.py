#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from typing import Optional

import torch

from torch import Tensor
from torch.nn import Module


DEFAULT_EPSILON = 2e-20


class Squeeze(Module):
    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.squeeze(self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class UnSqueeze(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Clamp(Module):
    def __init__(self, min_: float = -math.inf, max_: float = math.inf):
        super().__init__()
        self.min = min_
        self.max = max_

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, self.min, self.max)
