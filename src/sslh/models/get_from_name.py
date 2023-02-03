#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

from .mobilenet import MobileNetV1, MobileNetV2
from .mobilenet_rot import MobileNetV1Rot, MobileNetV2Rot
from .vgg import VGGLike
from .wideresnet28 import WideResNet28
from .wideresnet28_rot import WideResNet28Rot


def get_model_from_name(name: str, acronym: str, **kwargs) -> nn.Module:
    if name == "WideResNet28":
        model = WideResNet28(**kwargs)

    elif name == "MobileNetV1":
        model = MobileNetV1(**kwargs)

    elif name == "MobileNetV2":
        model = MobileNetV2(**kwargs)

    elif name == "VGGLike":
        model = VGGLike(**kwargs)

    # Models with rotation layer and a 'forward_rot()' method.
    elif name == "WideResNet28Rot":
        model = WideResNet28Rot(**kwargs)

    elif name == "MobileNetV1Rot":
        model = MobileNetV1Rot(**kwargs)

    elif name == "MobileNetV2Rot":
        model = MobileNetV2Rot(**kwargs)

    else:
        MODELS = (
            "WideResNet28",
            "MobileNetV1",
            "MobileNetV2",
            "WideResNet28Rot",
            "MobileNetV1Rot",
            "MobileNetV2Rot",
        )
        raise RuntimeError(f"Unknown model {name=}. Must be one of " f"{MODELS}.")

    return model
