#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable

import torch

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Sequential
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from sslh.nn.labels import MultiHot
from sslh.transforms.converters import ToTensor
from sslh.transforms.self_transforms.audio import get_self_transform_hvflips
from sslh.transforms.utils import compose_augment
from sslh.transforms.waveform.crop import Crop
from sslh.transforms.waveform.pad import Pad


N_CLASSES = 200


def get_transform_fsd50k(
    aug_cfg: DictConfig,
    n_mels: int = 64,
    n_time: int = 500,
    n_fft: int = 2048,
) -> Callable:
    # Get the augment pool
    pool = instantiate(aug_cfg)

    # Spectrogram shape : (channels, freq, time) = (1, 64, 501)
    waveform_length = 30  # seconds
    sample_rate = 44100
    target_length = sample_rate * waveform_length
    hop_length = sample_rate * waveform_length // n_time

    transform_to_spec = Sequential(
        Pad(target_length),
        MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        ),
        AmplitudeToDB(),
    )

    pre_transform = Sequential(
        ToTensor(dtype=torch.float),
        Crop(target_length),
    )
    post_transform = None

    augment = compose_augment(pool, transform_to_spec, pre_transform, post_transform)
    return augment


def get_target_transform_fsd50k(**kwargs) -> Callable:
    return MultiHot(N_CLASSES, torch.float)


def get_self_transform_fsd50k(**kwargs) -> Callable:
    return get_self_transform_hvflips()
