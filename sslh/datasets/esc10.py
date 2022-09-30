#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ESC-10 core classes and functions.
Developed by Léo Cances (leocances on Github).

Modified : Yes
    - typing & imports
"""

from typing import Optional, Tuple

from torch import nn, Tensor

from sslh.datasets.esc50 import cache_feature, ESC50Base, FOLDS, URL


class ESC10Base(ESC50Base):
    TARGET_MAPPER = {0: 0, 1: 1, 38: 2, 40: 3, 41: 4, 10: 5, 11: 6, 12: 7, 20: 8, 21: 9}

    def __init__(
        self,
        root: str = "data",
        folds: tuple = FOLDS,
        download: bool = False,
        transform: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(root, folds, download, transform)

        self.url = URL["esc10-10"]
        self.n_class = 10
        self.mapper = None  # Map the ESC-50 target to range(0, 10)

    def _load_metadata(self) -> None:
        super()._load_metadata()

        # Keep only the esc10-10 relevant files
        self._filenames = self._filenames[self._esc10s]
        self._targets = self._targets[self._esc10s]

    def __getitem__(self, index: int) -> Tuple[Tensor, int, int]:
        data, sampling_rate, target = super().__getitem__(index)
        return data, sampling_rate, ESC10Base.TARGET_MAPPER[target]


class ESC10(ESC10Base):
    @cache_feature
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        x, sr, y = super().__getitem__(index)
        return x, y
