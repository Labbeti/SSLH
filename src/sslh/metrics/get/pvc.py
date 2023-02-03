#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple

from torch import nn

from sslh.metrics.classification import AveragePrecision, CategoricalAccuracy, UAR


def get_metrics_pvc() -> Tuple[
    Dict[str, nn.Module], Dict[str, nn.Module], Dict[str, nn.Module]
]:
    train_metrics: dict[str, nn.Module] = {
        "acc": CategoricalAccuracy(),
    }
    val_metrics: dict[str, nn.Module] = {
        "acc": CategoricalAccuracy(),
    }
    val_metrics_stack = {
        "acc": CategoricalAccuracy(),
        "mAP": AveragePrecision(),
        "uar": UAR(),
    }
    return train_metrics, val_metrics, val_metrics_stack
