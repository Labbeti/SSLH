#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, List, Optional

from torch import Tensor

from sslh.metrics.base import Metric, IncrementalMetric
from sslh.metrics.incremental import IncrementalMean


class MetricWrapper(Metric):
    def __init__(
        self,
        callable_: Callable,
        use_input: bool = True,
        use_target: bool = True,
        reduce_fn: Optional[Callable] = None,
    ):
        """
        Wrapper of a callable function or class for comply with Metric typing_.

        :param callable_: The callable object to wrap.
        :param use_input: If True, the input_ argument will be passed as argument to the callable object wrapped.
        :param use_target: If True, the target argument will be passed as argument to the callable object wrapped.
        :param reduce_fn: The reduction function to apply.
        """
        super().__init__()
        self.callable_ = callable_
        self.reduce_fn = reduce_fn

        if use_input and use_target:
            self.sub_call = self._sub_call_both
        elif use_input:
            self.sub_call = self._sub_call_input
        elif use_target:
            self.sub_call = self._sub_call_target
        else:
            self.sub_call = self._sub_call_none

    def compute_score(self, pred, target):
        score = self.sub_call(pred, target)
        if self.reduce_fn is not None:
            score = self.reduce_fn(score)
        return score

    def _sub_call_both(self, input_: Tensor, target: Tensor) -> Tensor:
        return self.callable_(input_, target)

    def _sub_call_input(self, input_: Tensor, _target: Tensor) -> Tensor:
        return self.callable_(input_)

    def _sub_call_target(self, _input_: Tensor, target: Tensor) -> Tensor:
        return self.callable_(target)

    def _sub_call_none(self, _input_: Tensor, _target: Tensor) -> Tensor:
        return self.callable_()


class IncrementalWrapper(Metric):
    def __init__(
        self, metric: Metric, incremental_metric: IncrementalMetric = IncrementalMean()
    ):
        """
        Compute an incremental score (mean or std) of a metric.

        :param metric: The metric used to compute each score.
        :param incremental_metric: The incremental (continue) way to compute the mean or std.
        """
        super().__init__()
        self.metric = metric
        self.continue_metric = incremental_metric

    def compute_score(self, pred, target):
        score = self.metric(pred, target)
        self.continue_metric.add(score)
        return self.continue_metric.get_current()


class IncrementalListWrapper(Metric):
    def __init__(
        self,
        metric: Metric,
        incremental_metric_list: Optional[List[IncrementalMetric]] = None,
    ):
        """
        Compute a list of incremental scores (mean or std) of a metric.

        :param metric: The metric used to compute each score.
        :param incremental_metric_list: The list of incremental (continue) metrics for compute the mean or std.
        """
        super().__init__()
        self.metric = metric
        self.continue_metric_list = (
            incremental_metric_list if incremental_metric_list is not None else []
        )

    def compute_score(self, pred, target) -> list:
        score = self.metric(pred, target)
        for continue_metric in self.continue_metric_list:
            continue_metric.add(score)
        return [
            continue_metric.get_current()
            for continue_metric in self.continue_metric_list
        ]


class IncrementalList(IncrementalMetric):
    def __init__(self, incremental_list: List[IncrementalMetric]):
        super().__init__()
        self.incremental_list = incremental_list

    def reset(self):
        for incremental in self.incremental_list:
            incremental.reset()

    def add(self, value):
        for incremental in self.incremental_list:
            incremental.add(value)

    def is_empty(self) -> List[bool]:
        return [incremental.is_empty() for incremental in self.incremental_list]

    def get_current(self) -> list:
        return [incremental.get_current() for incremental in self.incremental_list]
