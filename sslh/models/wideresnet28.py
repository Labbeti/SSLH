#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sslh.models.wideresnet import WideResNet


class WideResNet28(WideResNet):
    """
    WideResNet-28 class. Expects an input of shape (bsize, 1, n_mels, time stamps).
    """

    def __init__(self, n_classes: int, width: int = 2, n_input_channels: int = 3):
        super().__init__(
            layers=[4, 4, 4],
            width=width,
            n_classes=n_classes,
            n_input_channels=n_input_channels,
        )
