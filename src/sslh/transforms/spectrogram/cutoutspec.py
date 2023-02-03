#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from typing import Tuple, Union

import torch

from torch import Tensor
from torch.nn import Module
from torch.distributions import Uniform

from sslh.transforms.base import SpectrogramTransform


class CutOutSpec(SpectrogramTransform):
    def __init__(
        self,
        freq_scales: Tuple[float, float] = (0.1, 0.5),
        time_scales: Tuple[float, float] = (0.1, 0.5),
        fill_value: Union[float, Tuple[float, float], str] = -100.0,
        fill_mode: Union[str, Module] = "constant",
        freq_dim: int = -2,
        time_dim: int = -1,
        p: float = 1.0,
    ) -> None:
        """
        CutOut transform for spectrogram tensors.

        Input must be of shape (..., freq, time).

        Example :

        >>> from sslh.transforms import CutOutSpec
        >>> spectrogram = ...
        >>> augment = CutOutSpec((0.5, 0.5), (0.5, 0.5))
        >>> # Remove 25% of the spectrogram values in a rectangle area
        >>> spectrogram_augmented = augment(spectrogram)

        :param freq_scales: The range of ratios for the frequencies dim. (default: (0.1, 0.5))
        :param time_scales: The range of ratios for the time steps dim. (default: (0.1, 0.5))
        :param fill_value: The value used for fill. Can be a range of values for sampling the fill value.
                This parameter is ignored if fill_mode is a custom Module.
                (default: -100.0)
        :param fill_mode: The fill mode.
                Can be 'constant', 'random' or a custom transform for the data delimited by the rectange.
                (default: 'constant')
        :param freq_dim: The dimension index of the spectrogram frequencies (default: -2)
        :param time_dim: The dimension index of the spectrogram time steps (default: -1)
        :param p: The probability to apply the transform. (default: 1.0)
        """
        if isinstance(fill_value, str) and fill_value != "min":
            raise ValueError(
                f"Invalid argument {fill_value=}. Expected float, tuple[float, float] or 'min'."
            )
        super().__init__(p=p)

        self.freq_scales = freq_scales
        self.time_scales = time_scales
        self.fill_value = fill_value
        self.fill_mode = fill_mode
        self.freq_dim = freq_dim
        self.time_dim = time_dim

        self._check_attributes()

    def process(self, data: Tensor) -> Tensor:
        if not isinstance(data, Tensor) or len(data.shape) < 2:
            raise RuntimeError(
                f"Input data must be a pytorch Tensor with at least 2 dimensions for CutOutSpec transform, "
                f"found {type(data)}"
                + (f" of shape {data.shape}" if hasattr(data, "shape") else "")
                + "."
            )

        # Prepare slices indexes for frequencies and time dimensions
        slices = [slice(None)] * len(data.shape)
        slices[self.freq_dim] = gen_range(data.shape[self.freq_dim], self.freq_scales)
        slices[self.time_dim] = gen_range(data.shape[self.time_dim], self.time_scales)

        data = data.clone()
        if self.fill_mode == "constant":
            data[slices] = self._gen_constant(data[slices])

        elif self.fill_mode == "random":
            data[slices] = self._gen_random(data[slices])

        elif isinstance(self.fill_mode, Module):
            data[slices] = self.fill_mode(data[slices])

        else:
            raise ValueError(
                f'Invalid fill_mode "{self.fill_mode}". '
                f'Must be one of "{("constant", "random")}" or a custom transform Module.'
            )

        return data

    def _gen_constant(self, data: Tensor) -> Tensor:
        if isinstance(self.fill_value, str):
            assert self.fill_value == "min"
            fill_value = data.min().item()
        elif isinstance(self.fill_value, float):
            fill_value = self.fill_value
        else:
            uniform = Uniform(*self.fill_value)
            fill_value = uniform.sample()
        return torch.full_like(data, fill_value)

    def _gen_random(self, data: Tensor) -> Tensor:
        if isinstance(self.fill_value, (float, str)):
            raise ValueError(
                "Invalid fill_value with random fill_mode. Please use a tuple of 2 floats for fill_value or use fill_mode='constant'."
            )
        else:
            uniform = Uniform(*self.fill_value)
            return uniform.sample(data.shape)

    def _check_attributes(self):
        if self.freq_dim == self.time_dim:
            raise ValueError(
                "Frequency dimension index cannot be the same than time dimension index."
            )

        if (
            not isinstance(self.fill_value, float)
            and not (isinstance(self.fill_value, tuple) and len(self.fill_value) == 2)
            and not isinstance(self.fill_value, str)
        ):
            raise ValueError(
                f'Invalid fill_value "{self.fill_value}", must be a float or a tuple of 2 floats.'
            )

        if self.fill_mode == "random" and isinstance(self.fill_value, (float, str)):
            raise ValueError(
                "Invalid fill_value with random fill_mode. Please use a tuple of 2 floats for fill_value or use "
                'fill_mode="constant".'
            )


def gen_range(size: int, scales: Tuple[float, float]) -> slice:
    """
    Generate an interval of size sampled from [size * scales[0], size * scales[1]].

    Example :

    >>> gen_range(size=100, scales=(0.5, 0.5))
    ... slice(10, 60)
    """
    cutout_size_min = math.ceil(scales[0] * size)
    cutout_size_max = max(math.ceil(scales[1] * size), cutout_size_min + 1)
    cutout_size = int(torch.randint(cutout_size_min, cutout_size_max, ()).item())

    cutout_start = torch.randint(0, max(size - cutout_size + 1, 1), ())
    cutout_end = cutout_start + cutout_size
    assert (
        cutout_end - cutout_start == cutout_size
    ), f"{cutout_end} - {cutout_start} != {cutout_size}"

    return slice(cutout_start, cutout_end)
