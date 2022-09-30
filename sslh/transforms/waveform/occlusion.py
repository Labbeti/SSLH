#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

import torch

from torch import Tensor

from sslh.transforms.base import WaveformTransform


class Occlusion(WaveformTransform):
    def __init__(
        self,
        scales: Tuple[float, float] = (0.1, 0.1),
        fill_value: float = 0.0,
        dim: int = -1,
        p: float = 1.0,
    ) -> None:
        """
        Occlusion transform.

        :param scales: The scale of the occlusion size. (default: 0.1)
        :param fill_value: The fill value for occlusion. (default: 0.0)
        :param dim: The dimension to apply the occlusion. (default: -1)
        :param p: The probability to apply the transform. (default: 1.0)
        """
        super().__init__(p=p)
        self.scales = scales
        self.fill_value = fill_value
        self.dim = dim

    def process(self, data: Tensor) -> Tensor:
        min_scale, max_scale = self.scales

        length = data.shape[self.dim]

        occlusion_min, occlusion_max = round(min_scale * length), round(
            max_scale * length
        )
        occlusion_max = max(occlusion_max, occlusion_min + 1)
        occlusion_size = int(
            torch.randint(low=occlusion_min, high=occlusion_max, size=()).item()
        )

        start_max = max(length - occlusion_size, 1)
        start = torch.randint(low=0, high=start_max, size=()).item()
        end = start + occlusion_size

        slices = [slice(None)] * len(data.shape)
        slices[self.dim] = slice(start, end)
        data = data.clone()
        data[slices] = self.fill_value
        return data
