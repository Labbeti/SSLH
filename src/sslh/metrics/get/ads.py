#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple

from torch import nn

from sslh.metrics.classification import (
    AveragePrecision,
    BinaryAccuracy,
    BCEMetric,
    DPrime,
    FScore,
    Recall,
    RocAuc,
    Precision,
)


def get_metrics_ads() -> Tuple[
    Dict[str, nn.Module], Dict[str, nn.Module], Dict[str, nn.Module]
]:
    threshold = 0.5
    thresholds = dict(threshold_input=threshold, threshold_target=threshold)

    train_metrics = {
        "f1": FScore(**thresholds),
        "bce": BCEMetric(),
    }
    val_metrics = {
        "f1": FScore(**thresholds),
        "bce": BCEMetric(),
    }
    val_metrics_stack = {
        "f1": FScore(**thresholds),
        "bce": BCEMetric(),
        "binacc": BinaryAccuracy(**thresholds),
        "recall": Recall(**thresholds),
        "precision": Precision(**thresholds),
        "mAP": AveragePrecision(),
        "auc": RocAuc(),
        "dprime": DPrime(),
    }
    return train_metrics, val_metrics, val_metrics_stack
