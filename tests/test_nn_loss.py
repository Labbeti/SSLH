#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from unittest import TestCase

import torch

from torch.nn import functional as F

from sslh.nn.loss import is_onehot_target


class TestCEVecTgts(TestCase):
    def test_is_onehot_targets_1(self) -> None:
        target = F.one_hot(torch.rand(16, 10).argmax(dim=-1), 10)
        self.assertTrue(is_onehot_target(target))

        target = torch.rand(16, 10).argmax(dim=-1)
        self.assertFalse(is_onehot_target(target))

        target = torch.rand(16, 10)
        self.assertFalse(is_onehot_target(target))

    def test_is_onehot_targets_2(self) -> None:
        target = torch.as_tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=torch.float)
        self.assertTrue(is_onehot_target(target))

        target = torch.as_tensor([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=torch.float)
        self.assertFalse(is_onehot_target(target))


if __name__ == "__main__":
    unittest.main()
