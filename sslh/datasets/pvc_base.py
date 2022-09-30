#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Primate Vocalisations Corpus (PVC) core classes and functions.
Developed by Léo Cances (leocances on Github).

Modified : Yes
    - typing & imports
"""

import os
import os.path as osp

from typing import Tuple

import torchaudio

from torch.utils.data.dataset import Dataset
from torch import Tensor


class COMPARE2021PRSBase(Dataset):
    CLASSES = ["background", "chimpanze", "geunon", "mandrille", "redcap"]

    def __init__(self, root: str = "data", subset: str = "train") -> None:
        SUBSETS = ("train", "test", "devel")
        if subset not in SUBSETS:
            raise ValueError(f"Invalid argument {subset=}. Expected one of {SUBSETS}.")

        self.root = root
        self.subset = subset

        self.subsets_info = self._load_csv()
        self.wav_dir = os.path.join(self.root, "ComParE2021_PRS", "dist", "wav")

        if not osp.isdir(root):
            raise RuntimeError(f'Invalid root dirpath "{root}".')

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        audio_name = self.subsets_info["audio_names"][idx]
        target = self.subsets_info["target"][idx]
        file_path = os.path.join(self.wav_dir, audio_name)

        waveform, sr = torchaudio.load(file_path)  # type: ignore

        return waveform, target

    def __len__(self):
        return len(self.subsets_info["audio_names"])

    def _to_cls_idx(self, target_str: str) -> int:
        if target_str == "?":
            return -1

        return COMPARE2021PRSBase.CLASSES.index(target_str)

    def _load_csv(self):
        def read_csv(path) -> dict:
            with open(path, "r") as f:
                lines = f.read().splitlines()
                lines = lines[1:]

            output = {
                "audio_names": [line.split(",")[0] for line in lines],
                "target": [self._to_cls_idx(line.split(",")[1]) for line in lines],
            }

            return output

        csv_root = os.path.join(self.root, "ComParE2021_PRS", "dist", "lab")
        if not osp.isdir(csv_root):
            raise RuntimeError(f'Invalid CSV root dirpath "{csv_root}".')

        if self.subset == "train":
            return read_csv(os.path.join(csv_root, "train.csv"))

        elif self.subset == "test":
            return read_csv(os.path.join(csv_root, "test.csv"))

        return read_csv(os.path.join(csv_root, "devel.csv"))
