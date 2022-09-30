#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC
from typing import Callable, List, Optional

import torch

from torch import Tensor

from sslh.metrics.base import IncrementalMetric


class IncrementalMean(IncrementalMetric):
    def __init__(self, *args: Tensor) -> None:
        """
        Compute the continue average of a values.
        """
        super().__init__()
        self._sum = torch.zeros(())
        self._counter = 0

        self.add_values(list(args))

    def reset(self) -> None:
        self._sum = torch.zeros(())
        self._counter = 0

    def add(self, value: Tensor):
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value)

        if self._sum is None:
            self._sum = value.clone()
            self._counter = 1
        else:
            self._sum += value
            self._counter += 1

    def is_empty(self) -> bool:
        return self._counter == 0

    def get_current(self) -> Optional[Tensor]:
        return self.get_mean()

    def get_mean(self) -> Optional[Tensor]:
        return (self._sum / self._counter) if self._counter > 0 else None

    def get_n_values_added(self) -> int:
        return self._counter

    def set_mean(self, mean: Tensor, counter: int):
        self._sum = mean * counter
        self._counter = counter


class IncrementalStd(IncrementalMetric):
    def __init__(self, *args: Tensor, unbiased: bool = False) -> None:
        """
        Compute the continue unbiased Standard Deviation (std).

        :param unbiased: If True, apply the bessel correction to std (like in the default std of pytorch).
                Otherwise return the classic std (like the default std of numpy).
                (default: False)
        """
        super().__init__()
        self._unbiased = unbiased
        self._items_sum = torch.zeros(())
        self._items_sq_sum = torch.zeros(())
        self._counter = 0

        self.add_values(list(args))

    def reset(self) -> None:
        self._items_sum = torch.zeros(())
        self._items_sq_sum = torch.zeros(())
        self._counter = 0

    def add(self, value: Tensor) -> None:
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value)

        if self._items_sum is None or self._items_sq_sum is None:
            self._items_sum = value.clone()
            self._items_sq_sum = value ** 2
            self._counter = 1
        else:
            self._items_sum += value
            self._items_sq_sum += value ** 2
            self._counter += 1

    def is_empty(self) -> bool:
        return self._counter == 0

    def get_current(self) -> Optional[Tensor]:
        return self.get_std()

    def get_std(self) -> Optional[Tensor]:
        if not self.is_empty():
            std = torch.sqrt(
                self._items_sq_sum / self._counter
                - (self._items_sum / self._counter) ** 2
            )
            if self._unbiased:
                std = (
                    std
                    * torch.scalar_tensor(self._counter / (self._counter - 1)).sqrt()
                )
            return std
        else:
            return None

    def get_n_values_added(self) -> int:
        return self._counter


class NBestsTracker(IncrementalMetric):
    def __init__(
        self,
        *args: Tensor,
        start_index: int = 0,
        is_better: Callable[[Tensor, Tensor], bool] = torch.gt,  # type: ignore
        n: int = 1,
    ):
        super().__init__()
        self._index = start_index
        self._is_better = is_better
        self._n = n

        self._start_index = start_index

        self._bests_list = []
        self._indexes_list = []

        self.add_values(list(args))

    def reset(self):
        self._bests_list = []
        self._indexes_list = []
        self._index = self._start_index

    def add(self, value: Tensor):
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value)

        insert_index = len(self._bests_list)
        for i, max_value in enumerate(self._bests_list):
            if self._check_is_better(value, max_value):
                insert_index = i
                break

        self._bests_list.insert(insert_index, value)
        self._indexes_list.insert(insert_index, self._index)

        while len(self._bests_list) > self._n:
            self._bests_list.pop()
            self._indexes_list.pop()

        self._index += 1

    def is_empty(self) -> bool:
        return len(self._bests_list) == 0

    def get_current(self) -> Optional[Tensor]:
        return self.get_max()

    def get_max(self) -> Optional[Tensor]:
        if self.is_empty():
            return None
        else:
            return torch.as_tensor(self._bests_list)

    def get_indexes_current(self) -> List[int]:
        return self._indexes_list

    def get_n_values_added(self) -> int:
        return self._index - self._start_index

    def _check_is_better(self, value: Tensor, best: Tensor) -> bool:
        return self._is_better(value, best)


class BestTracker(IncrementalMetric, ABC):
    def __init__(
        self,
        *args: Tensor,
        start_index: int = 0,
        start_best: Optional[Tensor] = None,
        start_index_best: int = -1,
    ):
        """
        Keep the best of the values stored.
        """
        super().__init__()
        self._index = start_index
        self._best = start_best
        self._index_best = start_index_best

        self._start_index = start_index

        self.add_values(list(args))

    def reset(self):
        self._best = None
        self._index_best = -1
        self._index = self._start_index

    def add(self, value: Tensor):
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value)

        if self._best is None or self._check_is_better(value, self._best):
            self._best = value.clone()
            self._index_best = self._index
        self._index += 1

    def is_empty(self) -> bool:
        return self._best is None

    def get_current(self) -> Optional[Tensor]:
        return self._best

    def get_index_current(self) -> int:
        return self._index_best

    def get_n_values_added(self) -> int:
        return self._index - self._start_index

    def _check_is_better(self, value: Tensor, best: Tensor) -> bool:
        raise NotImplementedError("Abstract method")


class MinTracker(BestTracker):
    def __init__(
        self,
        *args: Tensor,
        start_index: int = 0,
        start_min: Optional[Tensor] = None,
        start_index_min: int = -1,
    ):
        """
        Keep the minimum of the values stored.

        :param args: The optional values to add to MinTracker.
        :param start_index: The index of the start. (default: 0)
        :param start_min: The minimum value stored at start. (default: None)
        :param start_index_min: The index of the min value stored at start. (default: -1)
        """
        super().__init__(
            *args,
            start_index=start_index,
            start_best=start_min,
            start_index_best=start_index_min,
        )

    def get_min(self) -> Optional[Tensor]:
        return self.get_current()

    def get_index_min(self) -> int:
        return self.get_index_current()

    def _check_is_better(self, value: Tensor, best: Tensor) -> bool:
        return value.item() < best.item()


class MaxTracker(BestTracker):
    def __init__(
        self,
        *args: Tensor,
        start_index: int = 0,
        start_max: Optional[Tensor] = None,
        start_index_max: int = -1,
    ):
        """
        Keep the maximum of the values stored.

        :param args: The optional values to add to MaxTracker.
        :param start_index: The index of the start. (default: 0)
        :param start_min: The maximal value stored at start. (default: None)
        :param start_index_min: The index of the max value stored at start. (default: -1)
        """
        super().__init__(
            *args,
            start_index=start_index,
            start_best=start_max,
            start_index_best=start_index_max,
        )

    def get_max(self) -> Optional[Tensor]:
        return self.get_current()

    def get_index_max(self) -> int:
        return self.get_index_current()

    def _check_is_better(self, value: Tensor, best: Tensor) -> bool:
        return value.item() > best.item()


class BestTrackerBetterFunc(BestTracker):
    def __init__(
        self,
        *args: Tensor,
        start_index: int = 0,
        start_best: Optional[Tensor] = None,
        start_index_best: int = -1,
        is_better: Callable[[Tensor, Tensor], bool] = torch.gt,  # type: ignore
    ) -> None:
        super().__init__(
            *args,
            start_index=start_index,
            start_best=start_best,
            start_index_best=start_index_best,
        )
        self._is_better = is_better

    def _check_is_better(self, value: Tensor, best: Tensor) -> bool:
        return self._is_better(value, best)
