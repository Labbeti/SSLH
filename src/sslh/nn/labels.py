#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, Union

import numpy as np
import torch

from torch import nn, Tensor
from torch.nn import functional as F


class OneHot(nn.Module):
    def __init__(
        self,
        n_classes: int,
        smooth: Optional[float] = None,
        dtype: Optional[torch.dtype] = torch.float,
    ) -> None:
        """
        Convert label to one-hot encoding.

        :param n_classes: The number of classes in the dataset.
        :param smooth: The optional label smoothing coefficient parameter. (default: 0.0)
        """
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.dtype = dtype

    def forward(self, target: Union[int, list, np.ndarray, Tensor]) -> Tensor:
        target = torch.as_tensor(target)
        result = F.one_hot(target, self.n_classes)

        if self.smooth is not None:
            result = (1.0 - self.smooth) * result + self.smooth / self.n_classes

        if self.dtype is not None:
            result = result.to(dtype=self.dtype)

        return result


class MultiHot(nn.Module):
    def __init__(
        self,
        n_classes: int,
        dtype: torch.dtype = torch.bool,
        smooth: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.dtype = dtype
        self.smooth = smooth

    def forward(self, target: Union[int, list, np.ndarray, Tensor]) -> Tensor:
        if (isinstance(target, (int, float))) or (
            isinstance(target, (np.ndarray, Tensor)) and len(target.shape) == 0
        ):
            target = [[target]]
        elif (
            isinstance(target, list)
            and len(target) > 0
            and isinstance(target[0], (int, float))
        ) or (isinstance(target, (np.ndarray, Tensor)) and len(target.shape) == 1):
            target = [target]

        result = torch.zeros(len(target), self.n_classes, dtype=self.dtype)
        for i, indices in enumerate(target):
            for idx in indices:
                result[i, idx] = 1

        if self.smooth is not None:
            result = (1.0 - self.smooth) * result + self.smooth / self.n_classes

        result = result.squeeze()
        return result


class Thresholding(nn.Module):
    def __init__(
        self, threshold: Optional[float], bin_func: Callable = torch.ge
    ) -> None:
        """
        Convert label to multi-hot encoding.

        :param threshold: The threshold used to binarize the input.
                If None, the forward of this module will have no effect.
        :param bin_func: The comparison function used to binarize the Tensor. (default: torch.ge)
        """
        super().__init__()
        self.threshold = threshold
        self.bin_func = bin_func

    def forward(self, x: Tensor) -> Tensor:
        if self.threshold is not None:
            return self.bin_func(x, self.threshold).to(x.dtype)
        else:
            return x
