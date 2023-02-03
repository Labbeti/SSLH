#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

from typing import List, Optional, Tuple, Type

from sslh.transforms.base import ImageTransform
from sslh.transforms.image.ra_pools import RAND_AUGMENT_DEFAULT_POOL


class RandAugment(ImageTransform):
    def __init__(
        self,
        n_augm_apply: int = 1,
        magnitude: Optional[float] = 0.5,
        augm_pool: Optional[
            List[Tuple[Type[ImageTransform], Optional[Tuple[float, float]]]]
        ] = None,
        magnitude_policy: str = "random",
        p: float = 1.0,
    ):
        """
        Unofficial pytorch implementation of RandAugment.

        Original paper : https://arxiv.org/pdf/1909.13719.pdf

        :param n_augm_apply: The number of augmentations 'N' to apply on 1 image. (default: 1)
        :param magnitude: The magnitude 'M' used in RandAugment in range [0, 1].
                If magnitude_policy == 'random', this parameter is ignored.
                (default: 0.5)
        :param augm_pool: The list of augmentations with their optional range.
                If None, use the default pool.
                (default: None)
        :param magnitude_policy: The policy to apply for control magnitude of augmentations.
                Available policies are 'constant' and 'random'.
                (default: 'random')
        :param p: The probability to apply the transform. (default: 1.0)
        """
        assert magnitude is None or 0.0 <= magnitude <= 1.0
        assert magnitude_policy in ["constant", "random"]

        super().__init__(p=p)
        self._n_augm_apply = n_augm_apply
        self._magnitude = magnitude if magnitude is not None else random.random()
        self._augm_pool = (
            augm_pool if augm_pool is not None else RAND_AUGMENT_DEFAULT_POOL
        )
        self._magnitude_policy = magnitude_policy

        self._augms = _build_augms(self._augm_pool, self._magnitude)

    def process(self, x):
        if self._magnitude_policy == "random":
            new_magnitude = random.random()
            self.set_magnitude(new_magnitude)

        augms_to_apply = random.choices(self._augms, k=self._n_augm_apply)
        for augm in augms_to_apply:
            x = augm(x)
        return x

    def set_magnitude(self, magnitude: float):
        self._magnitude = magnitude if magnitude is not None else random.random()
        self._augms = _build_augms(self._augm_pool, self._magnitude)

    def get_magnitude(self) -> float:
        return self._magnitude

    def get_magnitude_policy(self) -> str:
        return self._magnitude_policy

    def get_n_augm_apply(self) -> int:
        return self._n_augm_apply


def _build_augms(
    augm_pool: List[Tuple[Type[ImageTransform], Optional[Tuple[float, float]]]],
    magnitude: float,
) -> List[ImageTransform]:
    augms = []
    for i, (augm_cls, range_) in enumerate(augm_pool):
        if range_ is not None:
            min_, max_ = range_
            if min_ >= 0.0 and max_ >= 0.0:
                augm_param = (max_ - min_) * magnitude + min_
            else:
                if random.random() < abs(min_) / (abs(min_) + abs(max_)):
                    augm_param = min_ * magnitude
                else:
                    augm_param = max_ * magnitude
            augm = augm_cls(augm_param)
        else:
            augm = augm_cls()
        augms.append(augm)
    return augms
