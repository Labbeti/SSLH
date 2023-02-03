#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

import torch


def random_rect(
    width_img: int,
    height_img: int,
    width_range: Tuple[float, float],
    height_range: Tuple[float, float],
) -> Tuple[int, int, int, int]:
    """
    Create a random rectangle inside an area defined by the limits (left, right, top, down).

    :param width_img: The maximal width.
    :param height_img: The maximal height.
    :param width_range: The width ratio range of the rectangle.
            Ex: (0.1, 0.5) => width is sampled from (0.1 * width, 0.5 * width).
    :param height_range: The height ratio range of the rectangle.
            Ex: (0.0, 0.9) => height is sampled from (0.0, 0.9 * height).
    :returns: The limits (left, right, top, down) of the rectangle created.
    """
    assert 0.0 <= width_range[0] <= width_range[1] <= 1.0
    assert 0.0 <= height_range[0] <= height_range[1] <= 1.0

    min_width = max(int(width_range[0] * width_img), 1)
    min_height = max(int(height_range[0] * height_img), 1)

    max_width = max(int(width_range[1] * width_img), min_width + 1)
    max_height = max(int(height_range[1] * height_img), min_height + 1)

    width = int(torch.randint(low=min_width, high=max_width, size=()).item())
    height = int(torch.randint(low=min_height, high=max_height, size=()).item())

    max_left = max(width_img - width, 1)
    max_top = max(height_img - height, 1)

    left = torch.randint(low=0, high=max_left, size=()).item()
    top = torch.randint(low=0, high=max_top, size=()).item()
    right = left + width
    down = top + height

    return left, right, top, down
