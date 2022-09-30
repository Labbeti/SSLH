#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional

import hydra
import torch

from omegaconf import DictConfig
from torch import nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from sslh.nn.labels import OneHot
from sslh.nn.tensor import UnSqueeze
from sslh.transforms.converters import ToTensor
from sslh.transforms.self_transforms.audio import (
    get_self_transform_hflip,
    get_self_transform_hvflips,
    get_self_transform_vflip,
)
from sslh.transforms.utils import compose_augment
from sslh.transforms.waveform.crop import Crop
from sslh.transforms.waveform.pad import Pad

N_CLASSES = 10


def get_transform_ubs8k(
    aug_cfg: DictConfig,
    n_mels: int = 64,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> Callable:
    pool = hydra.utils.instantiate(aug_cfg)

    # Spectrogram shape : (channels, freq, time) = (1, 64, 173)
    pad_length = 4  # (seconds), max length of UBS8K waveforms
    sample_rate = 22050
    target_length = sample_rate * pad_length

    transform_to_spec = nn.Sequential(
        Crop(target_length),
        Pad(target_length),
        MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        ),
        AmplitudeToDB(),
    )
    pre_transform = nn.Sequential(
        ToTensor(dtype=torch.float),
    )
    post_transform = UnSqueeze(dim=0)

    augment = compose_augment(pool, transform_to_spec, pre_transform, post_transform)
    return augment


def get_target_transform_ubs8k(smooth: Optional[float] = None) -> Callable:
    return OneHot(N_CLASSES, smooth, dtype=torch.float)


def get_self_transform_ubs8k(mode: str = "hvflips") -> Callable:
    if mode == "hvflips":
        return get_self_transform_hvflips()
    elif mode == "hflips":
        return get_self_transform_hflip()
    elif mode == "vflips":
        return get_self_transform_vflip()
    else:
        raise ValueError(f"Unknown self transform mode {mode=}.")
