#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Tuple

from torch.nn import Module


class UDAUnlabeledPreProcess(Module):
    """
    Compose transform_identity and transform_strong for unlabeled data.

    Note: (data, strong(data))
    """

    def __init__(self, transform_identity: Callable, transform_strong: Callable):
        super().__init__()
        self.transform_identity = transform_identity
        self.transform_strong = transform_strong

    def forward(self, data: Any) -> Tuple[Any, Any]:
        return self.transform_identity(data), self.transform_strong(data)
