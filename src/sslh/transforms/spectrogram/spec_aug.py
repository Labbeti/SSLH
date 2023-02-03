#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provides modules to use SpecAugment
BASED ON https://github.com/qiuqiangkong/torchlibrosa/blob/master/torchlibrosa/augmentation.py
MODIFIED: Yes (typing, spectrogram reshape, add probability p for specaugment, add check if audio is shorter than drop width)
"""

import random

import torch

from torch import nn, Tensor


class DropStripes(nn.Module):
    def __init__(self, dim: int, drop_width: int, stripes_num: int) -> None:
        """Drop stripes.

        Args:
                dim: int, dimension along which to drop
                drop_width: int, maximum width of stripes to drop
                stripes_num: int, how many stripes to drop
        """
        # Note: dim 2: time; dim 3: frequency
        if dim not in (2, 3):
            raise ValueError(f"Invalid argument dim={self.dim}. (expected 2 or 3)")
        if drop_width <= 0:
            raise ValueError(
                f"Invalid argument {drop_width=} in {self.__class__.__name__}. (expected a value > 0)"
            )

        super().__init__()
        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, spectro: Tensor) -> Tensor:
        """
        :param input: (batch_size, channels, time_steps, freq_bins) pr (channels, time_steps, freq_bins) tensor
        :return: Same shape as input.
        """
        is_single = spectro.ndim in (2, 3)
        if spectro.ndim == 2:
            # found (time_steps, freq_bins), so add bsize and channels dims
            spectro = spectro.unsqueeze_(dim=0).unsqueeze_(dim=0)
        elif spectro.ndim == 3:
            spectro = spectro.unsqueeze_(dim=0)

        if spectro.ndim != 4:
            raise ValueError(
                f"Invalid number of dimension for SpecAugmentation. (found {spectro.ndim=} but expected 2, 3 or 4 dims with shapes (bsize, channels, time_steps, freq_bins), the bsize and channels dims are optional.)"
            )

        if self.training:
            batch_size = spectro.shape[0]
            total_width = spectro.shape[self.dim]

            for n in range(batch_size):
                self._transform_slice(spectro[n], total_width)

        if is_single:
            spectro = spectro.squeeze_(0)

        return spectro

    def _transform_slice(self, spectro: Tensor, total_width: int) -> None:
        """inp: (channels, time_steps, freq_bins)"""
        # Add: If audio is empty, do nothing
        if total_width == 0:
            return None
        # Add: If audio is shorter than self.drop_width, clip drop width to total_width.
        drop_width = max(min(self.drop_width, total_width), 1)

        for _ in range(self.stripes_num):
            distance = int(torch.randint(low=0, high=drop_width, size=()).item())
            bgn = torch.randint(low=0, high=total_width - distance, size=())

            if self.dim == 2:
                spectro[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                spectro[:, :, bgn : bgn + distance] = 0
            else:
                raise ValueError(f"Invalid argument dim={self.dim}. (expected 2 or 3)")


class SpecAugmentation(nn.Module):
    def __init__(
        self,
        time_drop_width: int,
        time_stripes_num: int,
        freq_drop_width: int,
        freq_stripes_num: int,
        time_dim: int = 2,
        freq_dim: int = 3,
        inplace: bool = True,
        p: float = 1.0,
    ) -> None:
        """Spec augmentation.
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.

        Args:
            time_drop_width: int
            time_stripes_num: int
            freq_drop_width: int
            freq_stripes_num: int
        """
        super().__init__()
        self._inplace = inplace
        self._p = p

        self._time_dropper = DropStripes(
            dim=time_dim,
            drop_width=time_drop_width,
            stripes_num=time_stripes_num,
        )
        self._freq_dropper = DropStripes(
            dim=freq_dim,
            drop_width=freq_drop_width,
            stripes_num=freq_stripes_num,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.training and (self._p >= 1.0 or random.random() < self._p):
            if not self._inplace:
                x = x.clone()
            return self.transform(x)
        else:
            return x

    def transform(self, x: Tensor) -> Tensor:
        x = self._time_dropper(x)
        x = self._freq_dropper(x)
        return x
