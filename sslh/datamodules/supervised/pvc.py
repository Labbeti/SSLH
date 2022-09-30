#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from sslh.datasets.utils import TransformDataset
from sslh.datasets.pvc import (
    ComParE2021PRS,
    IterationBalancedSampler,
    class_balance_split,
)


N_CLASSES = 5


class PVCSupDataModule(LightningDataModule):
    def __init__(
        self,
        root: str = "data",
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        bsize: int = 256,
        n_workers: int = 4,
        drop_last: bool = False,
        pin_memory: bool = False,
        ratio: float = 1.0,
        n_train_steps: Optional[int] = 50000,
    ) -> None:
        """
        LightningDataModule of Primate Vocalization Corpus (PVC) for partial supervised trainings.

        Note: The subset of the dataset has the same class distribution.

        :param root: The root path of the dataset.
        :param train_transform: The optional transform to apply to train data. (default: None)
        :param val_transform: The optional transform to apply to validation data. (default: None)
        :param target_transform: The optional transform to apply to train and validation targets. (default: None)
        :param bsize: The batch size used for training and validation. (default: 30)
        :param n_workers: The number of workers for each dataloader. (default: 4)
        :param drop_last: If True, drop the last incomplete batch. (default: False)
        :param pin_memory: If True, pin the memory of dataloader. (default: False)
        :param ratio: The ratio of the subset len in [0, 1]. (default: 1.0)
        :param n_train_steps: The number of train steps for PVC.
                If None, the number will be set to the number of train labeled data.
                (default: 50000)
        """
        super().__init__()
        self.root = root
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.transform_test = val_transform
        self.target_transform = target_transform
        self.bsize_train = bsize
        self.bsize_val = bsize
        self.bsize_test = bsize
        self.n_workers = n_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.ratio = ratio

        self.n_train_steps = n_train_steps

        self.train_dataset_raw = None
        self.val_dataset_raw = None
        self.test_dataset_raw = None

        self.sampler_s = None
        self.example_input_array = None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset_raw = ComParE2021PRS(self.root, "train", transform=None)
            self.val_dataset_raw = ComParE2021PRS(self.root, "devel", transform=None)

            if self.ratio >= 1.0:
                indexes_s = list(range(len(self.train_dataset_raw)))
            else:
                indexes_s, _indexes_u = class_balance_split(
                    self.train_dataset_raw, self.ratio, None
                )

            if self.n_train_steps is None:
                n_max_samples_s = len(indexes_s) * self.bsize_train
            else:
                n_max_samples_s = self.n_train_steps * self.bsize_train

            self.sampler_s = IterationBalancedSampler(
                self.train_dataset_raw, indexes_s, n_max_samples_s
            )

            dataloader = self.val_dataloader()
            xs, ys = next(iter(dataloader))
            self.example_input_array = xs
            self.dims = tuple(xs.shape)

        elif stage == "test":
            # The 'test' subset is unlabeled, so we do not use it for now
            self.test_dataset_raw = None

    def train_dataloader(self) -> DataLoader:
        train_dataset_raw = self.train_dataset_raw
        if train_dataset_raw is None:
            raise RuntimeError(
                "Cannot call train_dataloader() if the datamodule is not setup."
            )
        train_dataset_raw = TransformDataset(
            train_dataset_raw, self.train_transform, index=0
        )
        train_dataset_raw = TransformDataset(
            train_dataset_raw, self.target_transform, index=1
        )

        loader = DataLoader(
            dataset=train_dataset_raw,
            batch_size=self.bsize_train,
            num_workers=self.n_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            sampler=self.sampler_s,
        )
        return loader

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
