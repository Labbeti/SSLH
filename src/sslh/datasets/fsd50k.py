#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os
import os.path as osp
import subprocess

from typing import Optional, Tuple

import torch
import torchaudio

from torch import nn, Tensor
from torch.utils.data.dataset import Dataset
from torchaudio.datasets.utils import download_url, extract_archive, validate_file


class FSD50KSubset(str):
    TRAIN = "train"
    VAL = "val"
    EVAL = "eval"
    DEV = "dev"


class FSD50K(Dataset):
    FOLDER = "fsd50k"
    N_CLASSES = 200
    SAMPLE_RATE = 44100

    def __init__(
        self,
        root: str = "data",
        subset: str = "dev",
        transform: Optional[nn.Module] = None,
        target_transform: Optional[nn.Module] = None,
        download: bool = False,
        verbose: int = 1,
    ) -> None:
        """
        Unofficial FreeSound Dataset 50k (FSD50K) pytorch dataset.
        Items are a tuple of audio waveform tensor and list of class indexes.

        >>> 'Dataset tree directories :'
        root/
        └── FSD50K/
                ├── FSD50K.dev_audio/
                │  	└── (40966 files, ~24GB)
                ├── FSD50K.doc/
                ├── FSD50K.eval_audio/
                │  	└── (10231 files, ~8.3GB)
                ├── FSD50K.ground_truth/
                │  	├── dev.csv
                │  	├── eval.csv
                │  	└── vocabulary.csv
                └── FSD50K.metadata/
                        └── collection/


        :param root: The dataset root directory path.
        :param subset: The name of the subset. Can be 'train', 'val', 'eval' or 'dev'.
                Note : The 'dev' subset is just the union of the 'train' and 'val' subsets.
        :param transform: The optional transform to apply to audio. (default: None)
        :param target_transform: The optional transform to apply the targets. (default: None)
        :param download: If True, download the dataset in the specified root.
                (default: False)
        :param verbose: The verbose level. (default: 1)
        """
        SUBSETS = ("dev", "train", "val", "eval")
        if subset not in SUBSETS:
            raise ValueError(f"Invalid argument {subset=}. Must be one of {SUBSETS}.")

        super().__init__()
        self._root = root
        self._subset = subset
        self._transform = transform
        self._target_transform = target_transform
        self._download = download
        self._verbose = verbose

        self._fnames = []
        self._targets = []

        if self._download:
            self._prepare_dataset()
        self._load_data()

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.get_audio(index), self.get_target(index)

    def __len__(self) -> int:
        return len(self._targets)

    def get_audio(self, index: int) -> Tensor:
        fpath = self.get_audio_fpath(index)
        audio, _sr = torchaudio.load(fpath)  # type: ignore
        if self._transform is not None:
            audio = self._transform(audio)
        return audio

    def get_target(self, index: int) -> Tensor:
        target = self._targets[index]
        if self._target_transform is not None:
            target = self._target_transform(target)
        return target

    def get_root_dir(self) -> str:
        return osp.join(self._root, self.FOLDER)

    def get_audio_fpath(self, index: int) -> str:
        return osp.join(
            self.get_root_dir(),
            SUBSET_INFO[self._subset]["audio_dir"],
            self._fnames[index],
        )

    def _prepare_dataset(self):
        if self._is_prepared():
            logging.info("Dataset already downloaded.")
            return

        root_dir = self.get_root_dir()
        files_info = URL_INFO["files"]
        hash_type = URL_INFO["hash_type"]

        if not osp.isdir(root_dir):
            os.makedirs(root_dir)

        # Download files
        for i, (fname, info) in enumerate(files_info.items()):
            url, hash_expected = info["url"], info["hash"]

            fpath = osp.join(root_dir, fname)

            if not osp.isfile(fpath):
                if self._verbose >= 2:
                    logging.info(f'Downloading file {i}/{len(files_info)} "{fname}"...')
                download_url(
                    url,
                    root_dir,
                    fname,
                    hash_expected,
                    hash_type,
                    progress_bar=self._verbose > 0,
                )
            else:
                with open(fpath, "rb") as file:
                    validated = validate_file(file, hash_expected, hash_type)
                if not validated:
                    raise RuntimeError(f'Invalid md5 checksum for file "{fname}".')

                if self._verbose >= 2:
                    logging.info(f'File "{fname}" already downloaded and verified.')

        # Merge & extract audio files
        if self._verbose >= 2:
            logging.info('Merging zip files for "FSD50K.dev_audio"...')
        fpath_split = osp.join(root_dir, "FSD50K.dev_audio.zip")
        fpath_unsplit = osp.join(root_dir, "FSD50K.dev_audio_unsplit.zip")
        if not osp.isfile(fpath_unsplit):
            subprocess.check_call(
                ["zip", "-s", "0", fpath_split, "--out", fpath_unsplit]
            )

        if self._verbose >= 2:
            logging.info('Merging zip files for "FSD50K.eval_audio"...')
        fpath_split = osp.join(root_dir, "FSD50K.eval_audio.zip")
        fpath_unsplit = osp.join(root_dir, "FSD50K.eval_audio_unsplit.zip")
        if not osp.isfile(fpath_unsplit):
            subprocess.check_call(
                ["zip", "-s", "0", fpath_split, "--out", fpath_unsplit]
            )

        fnames_to_extract = [
            "FSD50K.dev_audio_unsplit.zip",
            "FSD50K.eval_audio_unsplit.zip",
            "FSD50K.doc.zip",
            "FSD50K.ground_truth.zip",
            "FSD50K.metadata.zip",
        ]
        for fname in fnames_to_extract:
            fpath = osp.join(root_dir, fname)
            if self._verbose >= 2:
                logging.info(f'Extracting zip file "{fname}"...')
            extract_archive(fpath, root_dir)

        if self._verbose >= 2:
            logging.info("Dataset preparation done.")

    def _is_prepared(self) -> bool:
        root_dir = self.get_root_dir()

        dirs_exists = all(
            (
                osp.isdir(osp.join(root_dir, "FSD50K.dev_audio")),
                osp.isdir(osp.join(root_dir, "FSD50K.eval_audio")),
                osp.isdir(osp.join(root_dir, "FSD50K.ground_truth")),
            )
        )
        if not dirs_exists:
            return False

        return all(
            (
                len(os.listdir(osp.join(root_dir, "FSD50K.dev_audio")))
                == SUBSET_INFO["dev"]["n_samples"],
                len(os.listdir(osp.join(root_dir, "FSD50K.eval_audio")))
                == SUBSET_INFO["eval"]["n_samples"],
            )
        )

    def _load_data(self):
        if not self._is_prepared():
            raise RuntimeError(
                f'Dataset is not downloaded in directory "{self.get_root_dir()}" and download=False.'
            )

        root_dir = self.get_root_dir()

        fpath_vocabulary = osp.join(root_dir, "FSD50K.ground_truth", "vocabulary.csv")
        target_name_to_idx = {}
        with open(fpath_vocabulary, "r") as file:
            reader = csv.reader(file)
            for line in reader:
                target_name_to_idx[line[1]] = int(line[0])

        assert len(target_name_to_idx) == self.N_CLASSES

        fpath_ground_truth = osp.join(
            root_dir, SUBSET_INFO[self._subset]["ground_truth"]
        )

        fnames = []
        targets = []
        with open(fpath_ground_truth, "r") as file:
            reader = csv.DictReader(file)

            # Keys : fname,labels,mids,split
            for line in reader:
                fname = line["fname"]
                fname = f"{fname}.wav"

                labels = line["labels"]
                labels = labels.split(",")
                labels_indices = [target_name_to_idx[name] for name in labels]
                labels_indices = torch.as_tensor(labels_indices)

                is_in_subset = (
                    self._subset == FSD50KSubset.DEV
                    or self._subset == FSD50KSubset.EVAL
                    or (self._subset == FSD50KSubset.TRAIN and line["split"] == "train")
                    or (self._subset == FSD50KSubset.VAL and line["split"] == "val")
                )

                if is_in_subset:
                    fnames.append(fname)
                    targets.append(labels_indices)

                    # Check if file exists
                    fpath = osp.join(
                        self.get_root_dir(),
                        SUBSET_INFO[self._subset]["audio_dir"],
                        fname,
                    )
                    if not osp.isfile(fpath):
                        raise RuntimeError(f'File "{fname}" has not been downloaded.')

        self._fnames = fnames
        self._targets = targets
        assert len(self._fnames) == len(self._targets)


URL_INFO = {
    "hash_type": "md5",
    "files": {
        "FSD50K.dev_audio.z01": {
            "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01?download=1",
            "hash": "faa7cf4cc076fc34a44a479a5ed862a3",
        },
        "FSD50K.dev_audio.z02": {
            "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02?download=1",
            "hash": "8f9b66153e68571164fb1315d00bc7bc",
        },
        "FSD50K.dev_audio.z03": {
            "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03?download=1",
            "hash": "1196ef47d267a993d30fa98af54b7159",
        },
        "FSD50K.dev_audio.z04": {
            "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04?download=1",
            "hash": "d088ac4e11ba53daf9f7574c11cccac9",
        },
        "FSD50K.dev_audio.z05": {
            "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05?download=1",
            "hash": "81356521aa159accd3c35de22da28c7f",
        },
        "FSD50K.dev_audio.zip": {
            "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip?download=1",
            "hash": "c480d119b8f7a7e32fdb58f3ea4d6c5a",
        },
        "FSD50K.doc.zip": {
            "url": "https://zenodo.org/record/4060432/files/FSD50K.doc.zip?download=1",
            "hash": "3516162b82dc2945d3e7feba0904e800",
        },
        "FSD50K.eval_audio.z01": {
            "url": "https://zenodo.org/record/4060432/files/FSD50K.eval_audio.z01?download=1",
            "hash": "3090670eaeecc013ca1ff84fe4442aeb",
        },
        "FSD50K.eval_audio.zip": {
            "url": "https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip?download=1",
            "hash": "6fa47636c3a3ad5c7dfeba99f2637982",
        },
        "FSD50K.ground_truth.zip": {
            "url": "https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip?download=1",
            "hash": "ca27382c195e37d2269c4c866dd73485",
        },
        "FSD50K.metadata.zip": {
            "url": "https://zenodo.org/record/4060432/files/FSD50K.metadata.zip?download=1",
            "hash": "b9ea0c829a411c1d42adb9da539ed237",
        },
    },
}

SUBSET_INFO = {
    "dev": {
        "n_samples": 40966,
        "ground_truth": osp.join("FSD50K.ground_truth", "dev.csv"),
        "audio_dir": "FSD50K.dev_audio",
    },
    "eval": {
        "n_samples": 10231,
        "ground_truth": osp.join("FSD50K.ground_truth", "eval.csv"),
        "audio_dir": "FSD50K.eval_audio",
    },
    "train": {
        "n_samples": 36796,
        "ground_truth": osp.join("FSD50K.ground_truth", "dev.csv"),
        "audio_dir": "FSD50K.dev_audio",
    },
    "val": {
        "n_samples": 4170,
        "ground_truth": osp.join("FSD50K.ground_truth", "dev.csv"),
        "audio_dir": "FSD50K.dev_audio",
    },
}
