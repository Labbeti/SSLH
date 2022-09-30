#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Union

from torch import Tensor
from torch.distributions import Uniform

from sslh.transforms.base import Transform


class Fade(Transform):
    def __init__(self, factor: Union[float, Tuple[float, float]] = 0.5, p: float = 1.0):
        super().__init__(p=p)
        self.factor = factor
        self._check_attributes()

    def process(self, x: Tensor) -> Tensor:
        if isinstance(self.factor, float):
            factor = self.factor
        else:
            uniform = Uniform(*self.factor)
            factor = uniform.sample()

        min_ = x.min()
        x = min_ + (x - min_) * factor
        return x

    def _check_attributes(self):
        if not isinstance(self.factor, float) and not (
            isinstance(self.factor, tuple) and len(self.factor) == 2
        ):
            raise ValueError(
                f'Invalid factor "{self.fill_value}", must be a float or a tuple of 2 floats.'
            )
