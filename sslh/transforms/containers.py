#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

from abc import ABC
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch

from torch import nn

from sslh.transforms.base import Transform
from sslh.transforms.wrappers import TransformWrap


class Container(Transform, ABC):
    def __init__(self, *transforms: Callable, p: float = 1.0):
        super().__init__(p=p)
        self._transforms = list(transforms)

        for i, transform in enumerate(self._transforms):
            if not callable(transform):
                raise RuntimeError(
                    f'Cannot add non-callable object "{type(transform)}".'
                )
            if not isinstance(transform, nn.Module):
                transform = TransformWrap(transform)
            self.add_module(str(i), transform)

    def __getitem__(self, index: int) -> Callable:
        return self._transforms[index]

    def __len__(self) -> int:
        return len(self._transforms)

    def get_transforms(self) -> List[Callable]:
        return self._transforms

    def is_image_transform(self) -> bool:
        return all(
            isinstance(transform, Transform) and transform.is_image_transform()
            for transform in self._transforms
        )

    def is_waveform_transform(self) -> bool:
        return all(
            isinstance(transform, Transform) and transform.is_waveform_transform()
            for transform in self._transforms
        )

    def is_spectrogram_transform(self) -> bool:
        return all(
            isinstance(transform, Transform) and transform.is_spectrogram_transform()
            for transform in self._transforms
        )


class RandomChoice(Container):
    def __init__(
        self,
        *transforms: Callable,
        n_choices: Union[int, Tuple[int, int]] = 1,
        weights: Optional[Sequence[float]] = None,
        p: float = 1.0,
    ):
        """
        Select randomly k transforms in a list and apply them sequentially.

        An transform can be chosen multiple times if n_choices > 1. (with replacement)

        :param transforms: The list of transforms from we choose the apply a transform.
        :param n_choices: The number of transforms to choose.
                If tuple, it will be interpreted as a range [min,max[ for sampling the number of choices for each sample.
                (default: 1)
        :param weights: The probabilities to choose the transform. (default: None)
        :param p: The probability to apply the transform. (default: 1.0)
        """
        super().__init__(*transforms, p=p)
        self.n_choices = n_choices
        self.weights = weights

    def process(self, x: Any) -> Any:
        if isinstance(self.n_choices, tuple):
            n_choices_min, n_choices_max = self.n_choices
            n_choices = torch.randint(n_choices_min, n_choices_max, ()).item()
        else:
            n_choices = self.n_choices

        transforms = random.choices(
            self.get_transforms(), weights=self.weights, k=n_choices
        )
        for transform in transforms:
            x = transform(x)
        return x
