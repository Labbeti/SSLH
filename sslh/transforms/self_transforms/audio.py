#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Tuple

import torch

from torch import nn, Tensor
from torch.nn.functional import one_hot

from sslh.transforms.spectrogram.flips import HorizontalFlip, VerticalFlip


def get_self_transform_hvflips() -> Callable:
    transforms = [
        nn.Identity(),
        HorizontalFlip(),
        VerticalFlip(),
        nn.Sequential(HorizontalFlip(), VerticalFlip()),
    ]

    def self_transform(x: Tensor) -> Tuple[Tensor, Tensor]:
        bsize = len(x)
        transform_indexes = torch.randint(
            low=0, high=len(transforms), size=(bsize,), device=x.device
        )

        xr = torch.empty_like(x)
        for i, xi in enumerate(x):
            transform = transforms[transform_indexes[i]]
            xr[i] = transform(xi)

        yr = one_hot(transform_indexes, len(transforms))
        return xr, yr

    return self_transform


def get_self_transform_hflip() -> Callable:
    transforms = [
        nn.Identity(),
        HorizontalFlip(),
    ]

    def self_transform(x: Tensor) -> Tuple[Tensor, Tensor]:
        bsize = len(x)
        transform_indexes = torch.randint(
            low=0, high=len(transforms), size=(bsize,), device=x.device
        )

        xr = torch.empty_like(x)
        for i, xi in enumerate(x):
            transform = transforms[transform_indexes[i]]
            xr[i] = transform(xi)

        yr = one_hot(transform_indexes, len(transforms))
        return xr, yr

    return self_transform


def get_self_transform_vflip() -> Callable:
    transforms = [
        nn.Identity(),
        VerticalFlip(),
    ]

    def self_transform(x: Tensor) -> Tuple[Tensor, Tensor]:
        bsize = len(x)
        transform_indexes = torch.randint(
            low=0, high=len(transforms), size=(bsize,), device=x.device
        )

        xr = torch.empty_like(x)
        for i, xi in enumerate(x):
            transform = transforms[transform_indexes[i]]
            xr[i] = transform(xi)

        yr = one_hot(transform_indexes, len(transforms))
        return xr, yr

    return self_transform
