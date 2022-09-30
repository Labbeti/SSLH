#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

import numpy as np
import torch

from torch import Tensor

from sslh.datasets.utils import SizedDataset


def balanced_split(
    dataset: SizedDataset,
    n_classes: int,
    ratios: List[float],
    target_one_hot: bool = False,
) -> List[List[int]]:
    """
    Split dataset in list of indexes for each ratio.
    Also keep the original class distribution in every sub-dataset.

    :param dataset: The original dataset.
    :param n_classes: The number of classes in the original dataset.
    :param ratios: Ratios used to split the dataset. The sum must <= 1.
    :param target_one_hot: Consider labels as one-hot vectors. If False, consider labels as class indexes.
            (default: False)
    :return: A list of indexes for each ratios.
    """
    indexes_per_class = get_indexes_per_class(dataset, n_classes, target_one_hot)
    splits = split_indexes_per_class(indexes_per_class, ratios)

    def flat(lst_of_lst: List[List[int]]) -> List[int]:
        result = []
        for lst in lst_of_lst:
            result += lst
        return result

    indexes = [flat(split) for split in splits]
    return indexes


def get_indexes_per_class(
    dataset: SizedDataset,
    n_classes: int,
    target_one_hot: bool = False,
    label_index: int = 1,
) -> List[List[int]]:
    """
    Get class indexes from a Sized dataset with index of class as label.

    :param dataset: The mono-labeled sized dataset to iterate.
    :param n_classes: The number of classes in the dataset.
    :param target_one_hot: If True, convert each label as one-hot label encoding instead of class index. (default: False)
    :param label_index: TODO
    :return: The indexes per class in the dataset of size (n_classes, n_elem_in_class_i).
            Note: If the class distribution is not perfectly uniform, this return is not a complete matrix.
    """
    if not hasattr(dataset, "__len__"):
        raise RuntimeError("Dataset must have __len__() method for split indexes.")

    if hasattr(dataset, "targets") and isinstance(
        dataset.targets, (np.ndarray, Tensor, list)  # type: ignore
    ):
        targets = dataset.targets  # type: ignore
        targets = torch.as_tensor(targets)
        assert len(dataset) == len(
            targets
        ), "Dataset and targets must have the same len()."
    elif hasattr(dataset, "get_target") and callable(dataset.get_target):  # type: ignore
        targets = [torch.as_tensor(dataset.get_target(i)) for i in range(len(dataset))]  # type: ignore
        targets = torch.stack(targets)
    else:
        targets = [
            torch.as_tensor(dataset[i][label_index]) for i in range(len(dataset))
        ]
        targets = torch.stack(targets)

    if target_one_hot:
        targets = targets.argmax(dim=1)

    result = [
        torch.where(targets.eq(class_idx))[0].tolist() for class_idx in range(n_classes)
    ]
    return result


def split_indexes_per_class(
    indexes_per_class: List[List[int]],
    ratios: List[float],
) -> List[List[List[int]]]:
    """
    Split distinct indexes per class.

    Example :

    >>> split_indexes_per_class(indexes_per_class=[[1, 2], [3, 4], [5, 6]], ratios=[0.5, 0.5])
    ... [[[1], [3], [5]], [[2], [4], [6]]]

    :param indexes_per_class: List of indexes of each class.
    :param ratios: The ratios of each indexes split.
    :return: The indexes per ratio and per class of size (n_ratios, n_classes, n_indexes_in_ratio_and_class).
            Note: The return is not a tensor or ndarray because 'n_indexes_in_ratio_and_class' can be different for each
            ratio or class.
    """
    assert 0.0 <= sum(ratios) <= 1.0, "Ratio sum cannot be greater than 1.0."

    n_classes = len(indexes_per_class)
    n_ratios = len(ratios)

    indexes_per_ratio_per_class = [
        [[] for _ in range(n_classes)] for _ in range(n_ratios)
    ]

    current_starts = [0 for _ in range(n_classes)]
    for i, ratio in enumerate(ratios):
        for j, indexes in enumerate(indexes_per_class):
            current_start = current_starts[j]
            current_end = current_start + int(round(ratio * len(indexes)))
            sub_indexes = indexes[current_start:current_end]
            indexes_per_ratio_per_class[i][j] = sub_indexes
            current_starts[j] = current_end

    return indexes_per_ratio_per_class
