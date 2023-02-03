#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Callable, Final, Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sslh.datamodules.monolabel_split import balanced_split
from sslh.datamodules.utils import guess_num_workers_su
from sslh.datasets.utils import TransformDataset, NoLabelDataset
from sslh.datasets.gsc import SpeechCommands


N_CLASSES: Final = 35

logger = logging.getLogger(__name__)


class GSCSSLDataModule(LightningDataModule):
    def __init__(
        self,
        root: str = "data",
        train_transform_s: Optional[Callable] = None,
        train_transform_u: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        bsize_s: int = 64,
        bsize_u: int = 64,
        n_workers: int = 5,
        drop_last: bool = True,
        pin_memory: bool = False,
        ratio_s: float = 0.1,
        ratio_u: float = 0.9,
        duplicate_loader_s: bool = False,
        verbose: int = 0,
        download: bool = True,
    ) -> None:
        """
        LightningDataModule of GoogleSpeechCommands (GSC) for semi-supervised trainings.

        Note: The splits of the dataset has the same class distribution.

        :param root: The root path of the dataset.
        :param train_transform_s: The optional transform to apply to supervised train data. (default: None)
        :param train_transform_u: The optional transform to apply to unsupervised train data. (default: None)
        :param val_transform: The optional transform to apply to validation data. (default: None)
        :param target_transform: The optional transform to apply to train and validation targets. (default: None)
        :param bsize_s: The batch size used for supervised train data. (default: 64)
        :param bsize_u: The batch size used for unsupervised train data. (default: 64)
        :param n_workers: The number of workers the train dataloader. (default: 5)
        :param drop_last: If True, drop the last incomplete batch. (default: False)
        :param pin_memory: If True, pin the memory of dataloader. (default: False)
        :param ratio_s: The ratio of the supervised subset len in [0, 1]. (default: 0.1)
        :param ratio_u: The ratio of the unsupervised subset len in [0, 1]. (default: 0.9)
        :param duplicate_loader_s: If True, duplicate the supervised dataloader for DCT training. (default: False)
        :param download: If True, automatically download the dataset in the root directory. (default: True)
        """
        super().__init__()
        self.root = root
        self.train_transform_s = train_transform_s
        self.train_transform_u = train_transform_u
        self.val_transform = val_transform
        self.transform_test = val_transform
        self.target_transform = target_transform
        self.bsize_s = bsize_s
        self.bsize_u = bsize_u
        self.bsize_val = bsize_s + bsize_u
        self.bsize_test = bsize_s + bsize_u
        self.n_workers = n_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.ratio_s = ratio_s
        self.ratio_u = ratio_u
        self.duplicate_loader_s = duplicate_loader_s
        self.verbose = verbose

        self.download = download

        self.train_dataset_raw = None
        self.val_dataset_raw = None
        self.test_dataset_raw = None

        self.sampler_s = None
        self.sampler_u = None
        self.example_input_array = None

    def prepare_data(self, *args, **kwargs):
        if self.download:
            _ = SpeechCommands(self.root, "train", download=True)
            _ = SpeechCommands(self.root, "validation", download=True)
            _ = SpeechCommands(self.root, "testing", download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset_raw = SpeechCommands(self.root, "train", download=False)
            self.val_dataset_raw = SpeechCommands(
                self.root, "validation", download=False
            )

            # Setup split
            ratios = [self.ratio_s, self.ratio_u]
            indexes_s, indexes_u = balanced_split(
                dataset=self.train_dataset_raw,
                n_classes=N_CLASSES,
                ratios=ratios,
                target_one_hot=False,
            )
            self.sampler_s = SubsetRandomSampler(indexes_s)
            self.sampler_u = SubsetRandomSampler(indexes_u)

            dataloader = self.val_dataloader()
            xs, ys = next(iter(dataloader))
            self.example_input_array = xs
            self.dims = tuple(xs.shape)

        elif stage == "test":
            self.test_dataset_raw = SpeechCommands(self.root, "testing", download=False)

    def train_dataloader(self) -> Tuple[DataLoader, ...]:
        num_workers_s, num_workers_u = guess_num_workers_su(
            self.n_workers, self.bsize_s, self.bsize_u, self.verbose
        )

        train_dataset_raw = self.train_dataset_raw
        if train_dataset_raw is None:
            raise RuntimeError(
                "Cannot call train_dataloader() if the datamodule is not setup."
            )

        train_dataset_s = TransformDataset(
            train_dataset_raw, self.train_transform_s, index=0
        )
        train_dataset_s = TransformDataset(
            train_dataset_s, self.target_transform, index=1
        )

        train_dataset_u = TransformDataset(
            train_dataset_raw, self.train_transform_u, index=0
        )
        train_dataset_u = NoLabelDataset(train_dataset_u)

        loader_s = DataLoader(
            dataset=train_dataset_s,
            batch_size=self.bsize_s,
            num_workers=num_workers_s,
            sampler=self.sampler_s,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        loader_u = DataLoader(
            dataset=train_dataset_u,
            batch_size=self.bsize_u,
            num_workers=num_workers_u,
            sampler=self.sampler_u,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        if not self.duplicate_loader_s:
            loaders = loader_s, loader_u
        else:
            loaders = loader_s, loader_s, loader_u

        return loaders

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.val_dataset_raw
        if val_dataset is None:
            raise RuntimeError(
                "Cannot call val_dataloader() if the datamodule is not setup."
            )

        val_dataset = TransformDataset(val_dataset, self.val_transform, index=0)
        val_dataset = TransformDataset(val_dataset, self.target_transform, index=1)

        loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.bsize_val,
            num_workers=self.n_workers,
            drop_last=False,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        test_dataset = self.test_dataset_raw
        if test_dataset is None:
            raise RuntimeError(
                "Cannot call test_dataloader() if the datamodule is not setup."
            )

        test_dataset = TransformDataset(test_dataset, self.transform_test, index=0)
        test_dataset = TransformDataset(test_dataset, self.target_transform, index=1)

        loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.bsize_test,
            num_workers=self.n_workers,
            drop_last=False,
        )
        return loader
