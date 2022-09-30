#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp

from typing import Callable, List, Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sslh.datamodules.monolabel_split import balanced_split
from sslh.datamodules.utils import guess_folds
from sslh.datasets.ubs8k import UBS8KDataset
from sslh.datasets.utils import TransformDataset, NoLabelDataset


N_CLASSES = 10
FOLDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class UBS8KSSLDataModule(LightningDataModule):
    def __init__(
        self,
        root: str = "data",
        train_transform_s: Optional[Callable] = None,
        train_transform_u: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        bsize_s: int = 64,
        bsize_u: int = 64,
        n_workers_s: int = 2,
        n_workers_u: int = 3,
        drop_last: bool = True,
        pin_memory: bool = False,
        ratio_s: float = 0.1,
        ratio_u: float = 0.9,
        duplicate_loader_s: bool = False,
        download: bool = True,
        train_folds: Optional[List[int]] = None,
        val_folds: Optional[List[int]] = None,
        verbose: int = 0,
    ) -> None:
        """
        LightningDataModule of UrbanSound8K (UBS8K) for semi-supervised trainings.

        Note: The splits of the dataset has the same class distribution.

        :param root: The root path of the dataset.
        :param train_transform_s: The optional transform to apply to supervised train data. (default: None)
        :param train_transform_u: The optional transform to apply to unsupervised train data. (default: None)
        :param val_transform: The optional transform to apply to validation data. (default: None)
        :param target_transform: The optional transform to apply to train and validation targets. (default: None)
        :param bsize_s: The batch size used for supervised train data. (default: 64)
        :param bsize_u: The batch size used for unsupervised train data. (default: 64)
        :param n_workers_s: The number of workers for supervised train dataloader. (default: 2)
        :param n_workers_s: The number of workers for unsupervised train dataloader. (default: 3)
        :param drop_last: If True, drop the last incomplete batch. (default: False)
        :param pin_memory: If True, pin the memory of dataloader. (default: False)
        :param ratio_s: The ratio of the supervised subset len in [0, 1]. (default: 0.1)
        :param ratio_u: The ratio of the unsupervised subset len in [0, 1]. (default: 0.9)
        :param duplicate_loader_s: If True, duplicate the supervised dataloader for DCT training. (default: False)
        :param train_folds: The folds used for training.
                If None and val_folds is not None, then use the unused folds of validation.
                If both train_folds and val_folds are None, then the default folds are used:
                        [1, 2, 3, 4, 5, 6, 7, 8, 9] for train_folds and [10] for val_folds.
                (default: None)
        :param val_folds: The folds used for validation.
                If None and train_folds is not None, then use the unused folds of training.
                If both train_folds and val_folds are None, then the default folds are used:
                        [1, 2, 3, 4, 5, 6, 7, 8, 9] for train_folds and [10] for val_folds.
                (default: None)
        """
        if not osp.isdir(root):
            raise RuntimeError(f'Unknown dataset root dirpath "{root}" for UBS8K.')

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
        self.n_workers_s = n_workers_s
        self.n_workers_u = n_workers_u
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.ratio_s = ratio_s
        self.ratio_u = ratio_u
        self.duplicate_loader_s = duplicate_loader_s

        self.download = download
        self.train_folds, self.val_folds = guess_folds(
            train_folds, val_folds, FOLDS, verbose
        )

        self.train_dataset_raw = None
        self.val_dataset_raw = None
        self.test_dataset_raw = None

        self.sampler_s = None
        self.sampler_u = None
        self.example_input_array = None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset_raw = UBS8KDataset(self.root, folds=self.train_folds)
            self.val_dataset_raw = UBS8KDataset(self.root, folds=self.val_folds)

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
            self.test_dataset_raw = None

    def train_dataloader(self) -> Tuple[DataLoader, ...]:
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
            num_workers=self.n_workers_s,
            sampler=self.sampler_s,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        loader_u = DataLoader(
            dataset=train_dataset_u,
            batch_size=self.bsize_u,
            num_workers=self.n_workers_u,
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
                "Cannot call val_dataset() if the datamodule is not setup."
            )

        val_dataset = TransformDataset(val_dataset, self.val_transform, index=0)
        val_dataset = TransformDataset(val_dataset, self.target_transform, index=1)

        loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.bsize_val,
            num_workers=self.n_workers_s + self.n_workers_u,
            drop_last=False,
        )
        return loader

    def test_dataloader(self) -> None:
        return None
