#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from torch import Tensor
from torch.nn.functional import pad

from sslh.transforms.base import WaveformTransform


class Pad(WaveformTransform):
    def __init__(
        self,
        target_length: int,
        align: str = "left",
        fill_value: float = 0.0,
        dim: int = -1,
        mode: str = "constant",
        p: float = 1.0,
    ) -> None:
        """
        Example :

        >>> import torch; from torch import tensor
        >>> x = torch.ones(6)
        >>> zero_pad = Pad(10, align='left')
        >>> x_pad = zero_pad(x)
        ... tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

        :param target_length: The target length of the dimension.
        :param align: The alignment type. Can be 'left', 'right', 'center' or 'random'. (default: 'left')
        :param fill_value: The fill value used for constant padding. (default: 0.0)
        :param dim: The dimension to pad. (default: -1)
        :param mode: The padding mode. Can be 'constant', 'reflect', 'replicate' or 'circular'. (default: 'constant')
        :param p: The probability to apply the transform. (default: 1.0)
        """
        super().__init__(p=p)
        self.target_length = target_length
        self.align = align
        self.fill_value = fill_value
        self.dim = dim
        self.mode = mode

    def process(self, data: Tensor) -> Tensor:
        if self.align == "left":
            return self.pad_align_left(data)
        elif self.align == "right":
            return self.pad_align_right(data)
        elif self.align == "center":
            return self.pad_align_center(data)
        elif self.align == "random":
            return self.pad_align_random(data)
        else:
            raise ValueError(
                f'Unknown alignment "{self.align}". Must be one of {str(["left", "right", "center", "random"])}.'
            )

    def pad_align_left(self, x: Tensor) -> Tensor:
        # Note: pad_seq : [pad_left_dim_-1, pad_right_dim_-1, pad_left_dim_-2, pad_right_dim_-2, ...)
        idx = len(x.shape) - (self.dim % len(x.shape)) - 1
        pad_seq = [0 for _ in range(len(x.shape) * 2)]

        missing = max(self.target_length - x.shape[self.dim], 0)
        pad_seq[idx * 2 + 1] = missing

        x = pad(x, pad_seq, mode=self.mode, value=self.fill_value)
        return x

    def pad_align_right(self, x: Tensor) -> Tensor:
        idx = len(x.shape) - (self.dim % len(x.shape)) - 1
        pad_seq = [0 for _ in range(len(x.shape) * 2)]

        missing = max(self.target_length - x.shape[self.dim], 0)
        pad_seq[idx * 2] = missing

        x = pad(x, pad_seq, mode=self.mode, value=self.fill_value)
        return x

    def pad_align_center(self, x: Tensor) -> Tensor:
        idx = len(x.shape) - (self.dim % len(x.shape)) - 1
        pad_seq = [0 for _ in range(len(x.shape) * 2)]

        missing = max(self.target_length - x.shape[self.dim], 0)
        missing_left = missing // 2 + missing % 2
        missing_right = missing // 2

        pad_seq[idx * 2] = missing_left
        pad_seq[idx * 2 + 1] = missing_right

        x = pad(x, pad_seq, mode=self.mode, value=self.fill_value)
        return x

    def pad_align_random(self, x: Tensor) -> Tensor:
        idx = len(x.shape) - (self.dim % len(x.shape)) - 1
        pad_seq = [0 for _ in range(len(x.shape) * 2)]

        missing = max(self.target_length - x.shape[self.dim], 0)
        missing_left = torch.randint(low=0, high=missing + 1, size=()).item()
        missing_right = missing - missing_left

        pad_seq[idx * 2] = missing_left  # type: ignore
        pad_seq[idx * 2 + 1] = missing_right  # type: ignore

        x = pad(x, pad_seq, mode=self.mode, value=self.fill_value)
        return x

    def extra_repr(self) -> str:
        return (
            f"target_length={self.target_length}, "
            f"align={self.align}, "
            f"fill_value={self.fill_value}, "
            f"dim={self.dim}, "
            f"mode={self.mode}"
        )
