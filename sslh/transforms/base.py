#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

from abc import ABC

from torch import nn


class Transform(nn.Module, ABC):
    def __init__(self, p: float = 1.0) -> None:
        """
        Base class for all Transforms.

        :param p: The probability to apply the transform. (default: 1.0)
        """
        if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
            raise ValueError(
                f"Transform parameter p must be a float in range [0, 1]. (found {type(p)=} and value {p=}."
            )

        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p >= 1.0 or random.random() <= self.p:
            return self.process(x)
        else:
            return x

    def is_image_transform(self) -> bool:
        """
        :return: True if the transform can be applied to images.
        """
        return False

    def is_waveform_transform(self) -> bool:
        """
        :return: True if the transform can be applied to audio waveform signals.
        """
        return False

    def is_spectrogram_transform(self) -> bool:
        """
        :return: True if the transform can be applied to audio spectrogram signals.
        """
        return False

    def process(self, x):
        raise NotImplementedError("Abstract method")


class ImageTransform(Transform, ABC):
    def __init__(self, p: float = 1.0) -> None:
        super().__init__(p=p)

    def is_image_transform(self) -> bool:
        return True


class WaveformTransform(Transform, ABC):
    def __init__(self, p: float = 1.0) -> None:
        super().__init__(p=p)

    def is_waveform_transform(self) -> bool:
        return True


class SpectrogramTransform(Transform, ABC):
    def __init__(self, p: float = 1.0) -> None:
        super().__init__(p=p)

    def is_spectrogram_transform(self) -> bool:
        return True
