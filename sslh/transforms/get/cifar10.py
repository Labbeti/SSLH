#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, List, Optional

import hydra
import torch

from omegaconf import DictConfig
from torch import nn
from torchvision.transforms import ToTensor, Normalize

from sslh.nn.labels import OneHot
from sslh.nn.utils import Lambda
from sslh.transforms.self_transforms.image import get_self_transform_rotations
from sslh.transforms.utils import compose_augment


N_CLASSES = 10


def get_transform_cifar10(
    aug_cfg: DictConfig,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> Callable:
    if mean is None:
        mean = [0.4914009, 0.48215896, 0.4465308]
    if std is None:
        std = [0.24703279, 0.24348423, 0.26158753]

    pool = hydra.utils.instantiate(aug_cfg)

    transform_to_spec = None
    pre_transform = None
    post_transform = nn.Sequential(
        Lambda(ToTensor()),
        Normalize(mean=tuple(mean), std=tuple(std)),
    )

    augment = compose_augment(pool, transform_to_spec, pre_transform, post_transform)
    return augment


def get_target_transform_cifar10(smooth: Optional[float] = None) -> Callable:
    return OneHot(N_CLASSES, smooth, dtype=torch.float)


def get_self_transform_cifar10(**kwargs) -> Callable:
    return get_self_transform_rotations()
