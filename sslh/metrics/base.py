#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC
from typing import Any, Iterable

from torch import nn, Tensor


class Metric(nn.Module, ABC):
    """
    Base class for metric modules.

    Abstract methods:
            - compute_score(self, input_, target):
    """

    def __init__(self, input_to_cpu: bool = True, enable_graph: bool = False):
        super().__init__()
        self.input_to_cpu = input_to_cpu
        self.enable_graph = enable_graph

    def forward(self, pred, target):
        if self.input_to_cpu:
            if isinstance(pred, Tensor):
                pred = pred.cpu()
            if isinstance(target, Tensor):
                target = target.cpu()

        if not self.enable_graph:
            if isinstance(pred, Tensor):
                pred = pred.detach()
            if isinstance(target, Tensor):
                target = target.detach()

        score = self.compute_score(pred, target)
        return score

    def compute_score(self, pred, target):
        raise NotImplementedError("Abstract method")


class IncrementalMetric(nn.Module, ABC):
    """
    Base class for incremental metrics modules, which wrap a metric and compute a continue value on the scores.

    Abstract methods:
            - reset(self):
            - add(self, value: T):
            - get_current(self) -> Optional:
            - is_empty(self) -> bool:
    """

    def reset(self):
        """
        Reset the current incremental value.
        """
        raise NotImplementedError("Abstract method")

    def add(self, value):
        """
        Add a value to the incremental score.

        :param value: The value to add to the current incremental metric value.
        """
        raise NotImplementedError("Abstract method")

    def is_empty(self) -> bool:
        """
        :return: Return True if no value has been added to the incremental score.
        """
        raise NotImplementedError("Abstract method")

    def get_current(self) -> Any:
        """
        Get the current incremental score.

        :return: The current incremental metric value.
        """
        raise NotImplementedError("Abstract method")

    def add_values(self, values: Iterable):
        """
        Add a list of scores to the current incremental value.

        :param values: Add a of values to incremental metric.
        """
        for value in values:
            self.add(value)

    def forward(self, value) -> Any:
        """
        :param value: Add a value to the metric and returns the current incremental value.
        :return: The current incremental metric value.
        """
        self.add(value)
        return self.get_current()
