#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple

from torch import nn

from sslh.metrics.classification import CategoricalAccuracy
from sslh.metrics.wrappers import MetricWrapper
from sslh.nn.loss import NLLLossVecTargets


def get_metrics_ubs8k() -> Tuple[
    Dict[str, nn.Module], Dict[str, nn.Module], Dict[str, nn.Module]
]:
    train_metrics: Dict[str, nn.Module] = {
        "acc": CategoricalAccuracy(),
    }
    val_metrics = {
        "acc": CategoricalAccuracy(),
        "ce": MetricWrapper(NLLLossVecTargets()),
    }
    val_metrics_stack = {}
    return train_metrics, val_metrics, val_metrics_stack
