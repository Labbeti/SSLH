#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Union

import torch

from torch import Tensor, nn

config = [
    32,
    32,
    32,
    "M",
    64,
    64,
    "M",
    128,
]


def make_layers(
    cfg: List[Union[str, int]],
    batch_norm: bool = False,
    in_channels: int = 3,
):
    layers = []
    for v in cfg:
        if isinstance(v, str) and v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif isinstance(v, int):
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        else:
            raise ValueError(f"Invalid configuration value {v=}.")
    return nn.Sequential(*layers)


class VGGLike(nn.Module):
    """
    VGG-Like model used for FSD50K baseline.

    Paper FSD50K : https://arxiv.org/pdf/2010.00475.pdf
    """

    def __init__(
        self, n_classes: int, in_channels: int = 1, dropout_p: float = 0.0
    ) -> None:
        super().__init__()
        self.features = make_layers(config, True, in_channels)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_features = self.features(x)
        x1 = self.maxpool(x_features)
        x2 = self.avgpool(x_features)
        x = torch.cat((x1, x2), dim=1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
