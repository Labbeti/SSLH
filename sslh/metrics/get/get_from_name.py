#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple

from torch.nn import Module

from .ads import get_metrics_ads
from .cifar10 import get_metrics_cifar10
from .esc10 import get_metrics_esc10
from .fsd50k import get_metrics_fsd50k
from .gsc import get_metrics_gsc
from .pvc import get_metrics_pvc
from .ubs8k import get_metrics_ubs8k


def get_metrics(
    data_name: str,
) -> Tuple[Dict[str, Module], Dict[str, Module], Dict[str, Module]]:
    """
    Return the metrics used for a dataset.

    :param data_name: The name of the dataset. Can be one of: ('Audioset', 'CIFAR10', 'ESC10', 'GSC', 'PVC', 'UBS8K').
    :return: A tuple (train metrics dict, validation metrics dict, validation stack metrics dict).
            A 'stack' metric is a metric that must used all the predictions of the model as input (example: mAP metric).
            These metrics are computed only at the end of the training with the best model saved.
    """
    if data_name == "audioset":
        return get_metrics_ads()
    elif data_name in ("ssl_cifar10", "sup_cider10"):
        return get_metrics_cifar10()
    elif data_name in ("ssl_esc10", "sup_esc10"):
        return get_metrics_esc10()
    elif data_name in ("ssl_fsd50k", "sup_fsd50k"):
        return get_metrics_fsd50k()
    elif data_name in ("ssl_gsc", "sup_gsc"):
        return get_metrics_gsc()
    elif data_name in ("ssl_pvc", "sup_pvc"):
        return get_metrics_pvc()
    elif data_name in ("ssl_ubs8k", "sup_ubs8k"):
        return get_metrics_ubs8k()
    else:
        DATA_NAMES = ("audioset", "cifar10", "esc10", "fsd50k", "gsc", "pvc", "ubs8k")
        raise RuntimeError(
            f"Unknown argument {data_name=}. " f"Must be one of {DATA_NAMES}"
        )
