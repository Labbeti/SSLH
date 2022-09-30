#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Tuple, Union

import torch

from torch import Tensor
from torch.distributions import Uniform
from torchvision.transforms import RandomRotation

from sslh.transforms.base import ImageTransform
from sslh.transforms.image.utils import random_rect


class Normalize(ImageTransform):
    """
    Normalize an image in [0, 1].
    """

    def __init__(
        self,
        source_range: Tuple[Union[float, int], Union[float, int]] = (0, 255),
        target_range: Tuple[Union[float, int], Union[float, int]] = (0, 1),
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.source_range = source_range
        self.target_range = target_range

    def process(self, x: Tensor) -> Tensor:
        normalized = (x - self.source_range[0]) / (
            self.source_range[1] - self.source_range[0]
        )
        return (
            normalized * (self.target_range[1] - self.target_range[0])
            + self.target_range[0]
        )


class Standardize(ImageTransform):
    """
    Standardize image with a list of means and standard-deviations.
    """

    def __init__(
        self,
        means: Iterable[float],
        stds: Iterable[float],
        channel_dim: int = 0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.means = list(means)
        self.stds = list(stds)
        self.channel_dim = channel_dim

        if len(self.means) != len(self.stds):
            raise RuntimeError("Means and stds lists must have the same size.")

    def process(self, x: Tensor) -> Tensor:
        output = torch.empty_like(x)

        for channel_idx, (mean, std) in enumerate(zip(self.means, self.stds)):
            slices: list = [slice(None)] * len(x.shape)
            slices[self.channel_dim] = channel_idx
            output[slices] = (x[slices] - mean) / std
        return output


class Gray(ImageTransform):
    """
    Convert image to gray.
    """

    def __init__(self, dim_channel: int = 2, p: float = 1.0):
        super().__init__(p=p)
        self.dim_channel = dim_channel

    def process(self, x: Tensor) -> Tensor:
        n_channels = x.shape[self.dim_channel]
        output = x.mean(dim=self.dim_channel)
        output = output.repeat([n_channels] + [1] * (len(x.shape) - 1))
        output = output.permute(list(range(1, len(x.shape))) + [0])
        return output


class CutOutImg(ImageTransform):
    """
    Put gray value in an area randomly placed.
    """

    def __init__(
        self,
        width_scale_range: Tuple[float, float] = (0.1, 0.5),
        height_scale_range: Tuple[float, float] = (0.1, 0.5),
        fill_value: Union[float, int] = 0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.width_scale_range = width_scale_range
        self.height_scale_range = height_scale_range
        self.fill_value = fill_value

    def process(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 3

        width, height = x.shape[0], x.shape[1]
        left, right, top, down = random_rect(
            width, height, self.width_scale_range, self.height_scale_range
        )

        output = x.clone()
        slices = [slice(left, right), slice(top, down), slice(None)]
        output[slices] = self.fill_value
        return output


class UniColor(ImageTransform):
    """
    Delete 2 random channels in image for getting only 1 color.
    """

    def __init__(
        self, minimal_value: Union[float, int] = 0, dim_channel: int = 2, p: float = 1.0
    ):
        super().__init__(p=p)
        self.minimal_value = minimal_value
        self.dim_channel = dim_channel

    def process(self, x: Tensor) -> Tensor:
        output = torch.full_like(x, fill_value=self.minimal_value)
        channel_random = torch.randint(low=0, high=len(x.shape), size=())
        output[channel_random] = x.max(dim=self.dim_channel)
        return output


class Inversion(ImageTransform):
    """
    Invert pixel colors of an image.
    """

    def __init__(self, max_value: Union[float, int] = 255, p: float = 1.0):
        super().__init__(p=p)
        self.max_value = max_value

    def process(self, x: Tensor) -> Tensor:
        return self.max_value - x


class Rotation(ImageTransform):
    def __init__(
        self, degrees: Union[float, Tuple[float, float]] = 90.0, p: float = 1.0
    ):
        super().__init__(p=p)
        self.degrees = degrees if isinstance(degrees, tuple) else (degrees, degrees)
        self._rotation = RandomRotation(0)
        self._uniform = Uniform(low=self.degrees[0], high=self.degrees[1])

    def process(self, x: Tensor) -> Tensor:
        degree = self._uniform.sample().item()
        self._rotation.degrees = [degree, degree]
        return self._rotation(x)
