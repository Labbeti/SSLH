#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable

from omegaconf import DictConfig

from .ads import (
    get_transform_ads,
    get_target_transform_ads,
    get_self_transform_ads,
)
from .cifar10 import (
    get_transform_cifar10,
    get_target_transform_cifar10,
    get_self_transform_cifar10,
)
from .esc10 import (
    get_transform_esc10,
    get_target_transform_esc10,
    get_self_transform_esc10,
)
from .fsd50k import (
    get_transform_fsd50k,
    get_target_transform_fsd50k,
    get_self_transform_fsd50k,
)
from .gsc import (
    get_transform_gsc,
    get_target_transform_gsc,
    get_self_transform_gsc,
)
from .pvc import (
    get_transform_pvc,
    get_target_transform_pvc,
    get_self_transform_pvc,
)
from .ubs8k import (
    get_transform_ubs8k,
    get_target_transform_ubs8k,
    get_self_transform_ubs8k,
)


def get_transform(data_name: str, aug_cfg: DictConfig, **kwargs) -> Callable:
    """
    Returns the transform to apply to data for a specific dataset.

    Transform names available are :
            - identity (means no augment, but basic transforms like transform to spectrogram are returned)
            - weak (weak augment pool for MM, RMM and FM)
            - strong (strong augment pool for RMM, FM and UDA)

    :param data_name: The dataset of the transform.
    :param augment_name: The name of the transform.
    :return: The transform as Callable object.
    """
    if data_name in ("ssl_ads", "sup_ads"):
        return get_transform_ads(aug_cfg, **kwargs)
    elif data_name in ("ssl_cifar10", "sup_cider10"):
        return get_transform_cifar10(aug_cfg, **kwargs)
    elif data_name in ("ssl_esc10", "sup_esc10"):
        return get_transform_esc10(aug_cfg, **kwargs)
    elif data_name in ("ssl_fsd50k", "sup_fsd50k"):
        return get_transform_fsd50k(aug_cfg, **kwargs)
    elif data_name in ("ssl_gsc", "sup_gsc"):
        return get_transform_gsc(aug_cfg, **kwargs)
    elif data_name in ("ssl_pvc", "sup_pvc"):
        return get_transform_pvc(aug_cfg, **kwargs)
    elif data_name in ("ssl_ubs8k", "sup_ubs8k"):
        return get_transform_ubs8k(aug_cfg, **kwargs)
    else:
        DATASETS_NAMES = (
            "ssl_ads",
            "sup_ads",
            "ssl_cifar10",
            "sup_cifar10",
            "ssl_esc10",
            "sup_esc10",
            "ssl_fsd50k",
            "sup_fsd50k",
            "ssl_gsc",
            "sup_gsc",
            "ssl_pvc",
            "sup_pvc",
            "ssl_ubs8k",
            "sup_ubs8k",
        )
        raise ValueError(
            f"Unknown argument {data_name=}. Must be one of {DATASETS_NAMES}"
        )


def get_target_transform(data_name: str, **kwargs) -> Callable:
    if data_name in ("ssl_ads", "sup_ads"):
        return get_target_transform_ads(**kwargs)
    elif data_name in ("ssl_cifar10", "sup_cider10"):
        return get_target_transform_cifar10(**kwargs)
    elif data_name in ("ssl_esc10", "sup_esc10"):
        return get_target_transform_esc10(**kwargs)
    elif data_name in ("ssl_fsd50k", "sup_fsd50k"):
        return get_target_transform_fsd50k(**kwargs)
    elif data_name in ("ssl_gsc", "sup_gsc"):
        return get_target_transform_gsc(**kwargs)
    elif data_name in ("ssl_pvc", "sup_pvc"):
        return get_target_transform_pvc(**kwargs)
    elif data_name in ("ssl_ubs8k", "sup_ubs8k"):
        return get_target_transform_ubs8k(**kwargs)
    else:
        DATASETS_NAMES = (
            "ssl_ads",
            "sup_ads",
            "ssl_cifar10",
            "sup_cifar10",
            "ssl_esc10",
            "sup_esc10",
            "ssl_fsd50k",
            "sup_fsd50k",
            "ssl_gsc",
            "sup_gsc",
            "ssl_pvc",
            "sup_pvc",
            "ssl_ubs8k",
            "sup_ubs8k",
        )
        raise ValueError(
            f"Unknown argument {data_name=}. Must be one of {DATASETS_NAMES}"
        )


def get_self_transform(data_name: str, **kwargs) -> Callable:
    if data_name in ("ssl_ads", "sup_ads"):
        return get_self_transform_ads(**kwargs)
    elif data_name in ("ssl_cifar10", "sup_cider10"):
        return get_self_transform_cifar10(**kwargs)
    elif data_name in ("ssl_esc10", "sup_esc10"):
        return get_self_transform_esc10(**kwargs)
    elif data_name in ("ssl_fsd50k", "sup_fsd50k"):
        return get_self_transform_fsd50k(**kwargs)
    elif data_name in ("ssl_gsc", "sup_gsc"):
        return get_self_transform_gsc(**kwargs)
    elif data_name in ("ssl_pvc", "sup_pvc"):
        return get_self_transform_pvc(**kwargs)
    elif data_name in ("ssl_ubs8k", "sup_ubs8k"):
        return get_self_transform_ubs8k(**kwargs)
    else:
        DATASETS_NAMES = (
            "ssl_ads",
            "sup_ads",
            "ssl_cifar10",
            "sup_cifar10",
            "ssl_esc10",
            "sup_esc10",
            "ssl_fsd50k",
            "sup_fsd50k",
            "ssl_gsc",
            "sup_gsc",
            "ssl_pvc",
            "sup_pvc",
            "ssl_ubs8k",
            "sup_ubs8k",
        )
        raise ValueError(
            f"Unknown argument {data_name=}. Must be one of {DATASETS_NAMES}"
        )
