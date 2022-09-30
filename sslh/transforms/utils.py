#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, List, Optional, Tuple

from torch import nn

from sslh.nn.utils import Lambda
from sslh.transforms.containers import RandomChoice


def compose_augment(
    pool: List[dict[str, Any]],
    transform_to_spec: Optional[Callable],
    pre_transform: Optional[Callable],
    post_transform: Optional[Callable],
) -> Callable:
    """
    Compose augment pool with optional transform to spectrogram, pre-transform and post-transform.
    The augment pool will be automatically merged with a RandomChoice().

    If transform_to_spec is not None, the pool must contains tuples of (input_type, augment),
            where input_type is 'spectrogram' or 'waveform' and augment the callable object.
    The transform to spectrogram will be placed before or after each augment if the input type.

    Example 1 :

    >>> from torchaudio.transforms import MelSpectrogram
    >>> from sslh.nn.tensor import UnSqueeze
    >>> from sslh.transforms.converters import ToTensor
    >>> from sslh.transforms.waveform import Occlusion
    >>> from sslh.transforms.spectrogram import VerticalFlip
    >>> compose_augment(
    >>>		[('spectrogram', VerticalFlip()), ('waveform', Occlusion())],
    >>>		MelSpectrogram(),
    >>> 	ToTensor(),
    >>>		None
    >>>	)
    ... Sequential(
    ...		ToTensor(),
    ...		RandomChoice(
    ...			Sequential(
    ...				MelSpectrogram(),
    ...				VerticalFlip(),
    ...			),
    ...			Sequential(
    ...				Occlusion(),
    ...				MelSpectrogram(),
    ...			)
    ...		),
    ... )

    Example 2 :

    >>> from sslh.nn.tensor import UnSqueeze
    >>> from sslh.transforms.converters import ToTensor
    >>> from sslh.transforms.augments.flips import VerticalFlip
    >>> compose_augment([('image', VerticalFlip()]], None, ToTensor(), UnSqueeze(dim=0))
    ... Sequential(
    ...		ToTensor(),
    ...		VerticalFlip()
    ...		UnSqueeze(dim=0),
    ... )

    :param pool: The list of possible augments to apply.
    :param transform_to_spec: The optional transformation to spectrogram.
    :param pre_transform: The pre-transform to apply before augment & spectrogram.
    :param post_transform: The post-transform to apply after augment & spectrogram.
    :return: The augment pool composed as a Callable object.
    """
    pool_list_tuple = [(aug_info["type"], aug_info["aug"]) for aug_info in pool]
    pool_with_spec = add_transform_to_spec_to_pool(pool_list_tuple, transform_to_spec)
    augment = random_choice_pool(pool_with_spec)
    augment = add_pre_post_transforms(pre_transform, augment, post_transform)
    return augment


def add_transform_to_spec_to_pool(
    pool: List[Tuple[str, Callable]],
    transform_to_spec: Optional[Callable],
) -> List[Callable]:

    if transform_to_spec is None:
        return [augm for _, augm in pool]
    elif len(pool) == 0:
        return [transform_to_spec]

    pool_new = []
    for input_type, augm in pool:
        transforms = []

        if augm is not None:
            # Add transform to spectrogram before or after each augment depending on his internal type.
            if input_type == "waveform":
                transforms.append(augm)
                transforms.append(transform_to_spec)
            elif input_type == "spectrogram":
                transforms.append(transform_to_spec)
                transforms.append(augm)
            else:
                raise ValueError(
                    f'Invalid input type "{input_type}". Must be one of {("waveform", "spectrogram")}.'
                )
        else:
            transforms.append(transform_to_spec)

        if len(transforms) == 0:
            raise RuntimeError("Found an empty list of transforms.")
        elif len(transforms) == 1:
            pool_new.append(transforms[0])
        else:
            pool_new.append(nn.Sequential(*transforms))
    return pool_new


def random_choice_pool(pool: List[Callable]) -> Optional[Callable]:
    pool_filter = [transform for transform in pool if transform is not None]

    if len(pool_filter) == 0:
        return None
    elif len(pool_filter) == 1:
        return pool_filter[0]
    else:
        return RandomChoice(*pool_filter)


def add_pre_post_transforms(*transforms: Optional[Callable]) -> Callable:
    pool = [
        (transform if isinstance(transform, nn.Module) else Lambda(transform))
        for transform in transforms
        if transform is not None
    ]

    if len(pool) == 0:
        return nn.Identity()
    elif len(pool) == 1:
        return pool[0]
    else:
        return nn.Sequential(*pool)
