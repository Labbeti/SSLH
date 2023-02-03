#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable

import torch

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Sequential
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from sslh.nn.tensor import UnSqueeze
from sslh.transforms.converters import ToTensor
from sslh.transforms.self_transforms.audio import get_self_transform_hvflips
from sslh.transforms.utils import compose_augment


def get_transform_ads(
    aug_cfg: DictConfig,
    n_mels: int = 64,
    n_time: int = 500,
    n_fft: int = 2048,
    pre_computed_specs: bool = False,
) -> Callable:
    # Get the augment pool
    pool = instantiate(aug_cfg)

    # Spectrogram shape : (channels, freq, time) = (1, 64, 501)
    if pre_computed_specs:
        if not all(input_type == "spectrogram" for input_type, _ in pool):
            raise RuntimeError(
                "Use pre-computed spectrogram is True but augment pool contains waveform augments."
            )
        transform_to_spec = None
    else:
        waveform_length = 10  # seconds
        sample_rate = 32000
        hop_length = sample_rate * waveform_length // n_time
        transform_to_spec = Sequential(
            MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            ),
            AmplitudeToDB(),
        )

    pre_transform = ToTensor(dtype=torch.float)
    post_transform = UnSqueeze(dim=0)

    augment = compose_augment(pool, transform_to_spec, pre_transform, post_transform)
    return augment


def get_target_transform_ads(**kwargs) -> Callable:
    return ToTensor(dtype=torch.float)


def get_self_transform_ads(**kwargs) -> Callable:
    return get_self_transform_hvflips()
