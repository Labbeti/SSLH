#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

import torch

from torch import nn, Tensor
from torch.distributions.uniform import Uniform

from sslh.transforms.base import Transform


class AdditiveNoise(Transform):
    def __init__(
        self, snr_db: float, clamp_max: Union[float, str] = "auto", p: float = 1.0
    ):
        super().__init__(p=p)
        self.snr_db = snr_db
        self.clamp_max = clamp_max

    def process(self, x: Tensor) -> Tensor:
        if isinstance(self.clamp_max, float):
            clamp_max = self.clamp_max
        elif isinstance(self.clamp_max, str):
            assert self.clamp_max == "auto"
            clamp_max = float(x.max())
        else:
            raise TypeError(f"Invalid argument type {self.clamp_max=}.")

        x = x + gen_noise(x, self.snr_db)
        if clamp_max is not None:
            x = torch.clamp(x, max=clamp_max)
        return x


class SubtractiveNoise(Transform):
    def __init__(
        self, snr_db: float, clamp_min: Union[float, str, None] = "auto", p: float = 1.0
    ):
        super().__init__(p=p)
        self.snr_db = snr_db
        self.clamp_min = clamp_min

    def process(self, x: Tensor) -> Tensor:
        if isinstance(self.clamp_min, float):
            clamp_min = self.clamp_min
        elif isinstance(self.clamp_min, str):
            assert self.clamp_min == "auto"
            clamp_min = float(x.max())
        else:
            raise TypeError(f"Invalid argument type {self.clamp_min=}.")

        x = x - gen_noise(x, self.snr_db)
        if clamp_min is not None:
            x = torch.clamp(x, min=clamp_min)
        return x


def gen_noise(x: Tensor, snr_db: float) -> Tensor:
    mean_sq_x = (x ** 2).mean()
    snr = 10 ** (snr_db / 10)
    high = torch.sqrt(3.0 * mean_sq_x / snr)
    uniform = Uniform(low=0, high=high)
    noise = uniform.sample(x.shape)
    return noise


class GaussianNoise(nn.Module):
    def __init__(self, snr_db: float = 15.0) -> None:
        super().__init__()
        self.snr_db = snr_db

    def forward(self, x: Tensor) -> Tensor:
        noise = torch.rand_like(x) * self.snr_db + self.snr_db
        return x + noise
