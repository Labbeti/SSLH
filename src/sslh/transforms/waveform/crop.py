#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from torch import Tensor

from sslh.transforms.base import WaveformTransform


class Crop(WaveformTransform):
    def __init__(
        self, target_length: int, align: str = "left", dim: int = -1, p: float = 1.0
    ) -> None:
        super().__init__(p=p)
        self.target_length = target_length
        self.align = align
        self.dim = dim

    def process(self, data: Tensor) -> Tensor:
        if self.align == "center":
            return self.crop_align_center(data)
        elif self.align == "left":
            return self.crop_align_left(data)
        elif self.align == "random":
            return self.crop_align_random(data)
        elif self.align == "right":
            return self.crop_align_right(data)
        else:
            raise ValueError(
                f'Unknown alignment "{self.align}". Must be one of {str(["left", "right", "center", "random"])}.'
            )

    def crop_align_center(self, data: Tensor) -> Tensor:
        if data.shape[self.dim] > self.target_length:
            diff = data.shape[self.dim] - self.target_length
            start = diff // 2 + diff % 2
            end = start + self.target_length
            slices = [slice(None)] * len(data.shape)
            slices[self.dim] = slice(start, end)
            data = data[slices]
            data = data.contiguous()
        return data

    def crop_align_left(self, data: Tensor) -> Tensor:
        if data.shape[self.dim] > self.target_length:
            slices = [slice(None)] * len(data.shape)
            slices[self.dim] = slice(self.target_length)
            data = data[slices]
            data = data.contiguous()
        return data

    def crop_align_random(self, data: Tensor) -> Tensor:
        if data.shape[self.dim] > self.target_length:
            diff = data.shape[self.dim] - self.target_length
            start = torch.randint(low=0, high=diff, size=()).item()
            end = start + self.target_length
            slices = [slice(None)] * len(data.shape)
            slices[self.dim] = slice(start, end)
            data = data[slices]
            data = data.contiguous()
        return data

    def crop_align_right(self, data: Tensor) -> Tensor:
        if data.shape[self.dim] > self.target_length:
            start = data.shape[self.dim] - self.target_length
            slices = [slice(None)] * len(data.shape)
            slices[self.dim] = slice(start, None)
            data = data[slices]
            data = data.contiguous()
        return data

    def extra_repr(self) -> str:
        return (
            f"target_length={self.target_length}, "
            f"align={self.align}, "
            f"dim={self.dim}"
        )
