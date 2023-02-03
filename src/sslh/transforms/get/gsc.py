#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional

import torch

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from sslh.nn.labels import OneHot
from sslh.transforms.self_transforms.audio import (
    get_self_transform_hflip,
    get_self_transform_hvflips,
    get_self_transform_vflip,
)
from sslh.transforms.utils import compose_augment
from sslh.transforms.waveform.pad import Pad

N_CLASSES = 35


def get_transform_gsc(
    aug_cfg: DictConfig,
    n_mels: int = 64,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> Callable:
    pool = instantiate(aug_cfg)

    # Spectrogram shape : (channels, freq, time) = (1, 64, 32)
    waveform_length = 1  # seconds
    sample_rate = 16000
    target_length = sample_rate * waveform_length

    transform_to_spec = nn.Sequential(
        Pad(target_length),
        MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        ),
        AmplitudeToDB(),
    )
    pre_transform = None
    post_transform = None

    augment = compose_augment(pool, transform_to_spec, pre_transform, post_transform)
    return augment


def get_target_transform_gsc(smooth: Optional[float] = None) -> Callable:
    return OneHot(N_CLASSES, smooth, dtype=torch.float)


def get_self_transform_gsc(mode: str = "hvflips") -> Callable:
    if mode == "hvflips":
        return get_self_transform_hvflips()
    elif mode == "hflips":
        return get_self_transform_hflip()
    elif mode == "vflips":
        return get_self_transform_vflip()
    else:
        raise ValueError(f"Unknown self transform mode {mode=}.")
