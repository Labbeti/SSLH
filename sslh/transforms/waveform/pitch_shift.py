#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import warnings

from typing import Tuple

import librosa
import torch

from torch import nn, Tensor


class PitchShiftRandom(nn.Module):
    def __init__(
        self, sr: int = 32000, steps: Tuple[float, float] = (-3, 3), p: float = 1.0
    ) -> None:
        super().__init__()
        self.sr = sr
        self.steps = steps
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p >= 1.0 or random.random() <= self.p:
            return self.transform(x)
        else:
            return x

    def transform(self, x: Tensor) -> Tensor:
        device = x.device
        if x.is_cuda:
            warnings.warn(
                "Using PitchShiftRandom on CUDA tensor is slow because it requires to move them to CPU."
            )
            x = x.cpu()

        x = x.numpy()
        n_steps = random.uniform(*self.steps)
        x = librosa.effects.pitch_shift(x, sr=self.sr, n_steps=n_steps)  # type: ignore
        x = torch.from_numpy(x).to(device)
        return x
