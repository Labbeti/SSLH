#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from torch import Tensor
from torch.nn import Linear

from sslh.models.wideresnet import BasicBlock
from sslh.models.wideresnet28 import WideResNet28


class WideResNet28Rot(WideResNet28):
    """
    WideResNet-28 class with rotation layer. Expects an input of shape (bsize, 1, n_mels, time stamps).
    """

    def __init__(
        self, n_classes: int, rot_size: int, width: int = 2, n_input_channels: int = 3
    ):
        super().__init__(
            width=width, n_classes=n_classes, n_input_channels=n_input_channels
        )
        self.fc_rot = Linear(64 * width * BasicBlock.expansion, rot_size)

    def forward_rot(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_rot(x)

        return x
