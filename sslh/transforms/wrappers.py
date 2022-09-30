#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, List, Optional, Tuple, Union

import torch

from PIL import Image
from torch import Tensor

from sslh.transforms.base import ImageTransform, Transform
from sslh.transforms.converters import ToPIL, ToTensor


class RandomApplyWrap(Transform):
    def __init__(self, callable_: Callable, p: float = 1.0):
        super().__init__(p=p)
        self.callable = callable_

    def process(self, x):
        return self.callable(x)


class ProcessWrap(Transform):
    def __init__(
        self,
        transform: Optional[Transform],
        pre_convert: Optional[Callable],
        post_convert: Optional[Callable],
        p: float = 1.0,
    ):
        """
        :param transform: The transform to wrap.
        :param pre_convert: The function called before the transform.
        :param post_convert: The function called after the transform.
        :param p: The probability to apply the transform. (default: 1.0)
        """
        super().__init__(p=p)
        self.transform = transform
        self.pre_convert = pre_convert
        self.post_convert = post_convert

    def process(self, x: Any) -> Any:
        for callable_ in self.get_transforms():
            x = callable_(x)
        return x

    def is_image_transform(self) -> bool:
        return self.transform.is_image_transform()

    def is_spectrogram_transform(self) -> bool:
        return self.transform.is_spectrogram_transform()

    def is_waveform_transform(self) -> bool:
        return self.transform.is_waveform_transform()

    def unwrap(self) -> Optional[Transform]:
        return self.transform

    def get_transforms(self) -> List[Callable]:
        return [
            callable_
            for callable_ in (self.pre_convert, self.transform, self.post_convert)
            if callable_ is not None
        ]


class PILInternalWrap(ProcessWrap):
    def __init__(
        self, pil_transform: ImageTransform, mode: Optional[str] = "RGB", p: float = 1.0
    ):
        """
        Class that convert tensor to PIL image internally for apply PIL transforms.
        Tensors images must have the shape (width, height, 3).

        :param pil_transform: The PIL transform to wrap.
        :param mode: The PIL image mode of the image. (default: 'RGB')
        :param p: The probability to apply the transform. (default: 1.0)
        """
        super().__init__(
            transform=pil_transform,
            pre_convert=ToPIL(mode=mode),
            post_convert=ToTensor(),
            p=p,
        )

    def process(self, x: Tensor) -> Tensor:
        return super().process(x)


class TensorInternalWrap(ProcessWrap):
    def __init__(self, pil_transform: ImageTransform, p: float = 1.0):
        """
        Class that convert PIL image to tensor internally for apply tensor transforms.
        Tensors images will have the shape (width, height, 3).

        :param pil_transform: The tensor transform to wrap.
        :param p: The probability to apply the transform. (default: 1.0)
        """
        super().__init__(
            transform=pil_transform,
            pre_convert=ToTensor(),
            post_convert=ToPIL(mode=None),
            p=p,
        )

    def process(self, x: Image.Image) -> Image.Image:
        self.post_convert.mode = x.mode
        return self.post_convert(self.transform(self.pre_convert(x)))


class TransformWrap(Transform):
    def __init__(
        self,
        callable_: Callable,
        image_transform: bool = False,
        waveform_transform: bool = False,
        spectrogram_transform: bool = False,
        p: float = 1.0,
    ):
        """
        Wrap a callable object to Transform.
        Useful for keep correct typing_ and adding transform type (image, waveform or spectrogram) information.

        :param callable_: The callable object to wrap.
        :param image_transform: Indicate if the object wrapped is an image transform or not. (default: False)
        :param waveform_transform: Indicate if the object wrapped is an waveform transform or not. (default: False)
        :param spectrogram_transform: Indicate if the object wrapped is an spectrogram transform or not. (default: False)
        :param p: The probability to apply the transform. (default: 1.0)
        """
        super().__init__(p=p)
        self.callable_ = callable_
        self.image_transform = image_transform
        self.waveform_transform = waveform_transform
        self.spectrogram_transform = spectrogram_transform

    def process(self, x: Tensor) -> Tensor:
        return self.callable_(x)

    def is_image_transform(self) -> bool:
        return self.image_transform

    def is_waveform_transform(self) -> bool:
        return self.waveform_transform

    def is_spectrogram_transform(self) -> bool:
        return self.spectrogram_transform

    def unwrap(self) -> Callable:
        return self.callable_

    def extra_repr(self) -> str:
        if hasattr(self.callable_, "__name__"):
            return self.callable_.__name__
        elif hasattr(self.callable_, "__class__"):
            return self.callable_.__class__.__name__
        else:
            return ""


class Duplicate(Transform):
    def __init__(
        self, transform: Transform, n: Union[int, Tuple[int, int]], p: float = 1.0
    ):
        super().__init__(p=p)
        self.transform = transform
        self.n = n

    def process(self, x: Any) -> Any:
        if isinstance(self.n, tuple):
            n = torch.randint(*self.n)
        else:
            n = self.n

        for _ in range(n):
            x = self.transform(x)
        return x

    def is_image_transform(self) -> bool:
        return self.transform.is_image_transform()

    def is_spectrogram_transform(self) -> bool:
        return self.transform.is_spectrogram_transform()

    def is_waveform_transform(self) -> bool:
        return self.transform.is_waveform_transform()

    def unwrap(self) -> Transform:
        return self.transform
