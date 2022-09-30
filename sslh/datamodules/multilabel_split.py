#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import random
import tqdm

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch

from torch import Tensor

from sslh.datasets.utils import SizedDataset


def balanced_split(
    dataset: SizedDataset,
    n_classes: int,
    target_type: str,
    ratio_s: float,
    ratio_u: Optional[float],
    shuffle: bool = True,
    verbose: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    :param dataset: TODO
    :param n_classes: TODO
    :param target_type: 'indexes' or 'multihot'.
    :param ratio_s: float in [0, 1].
    :param ratio_u: float in [0, 1]. If None, it will be set to (1.0 - ratio_s).
    :param shuffle: TODO
    :param verbose: TODO
    """
    if ratio_u is None:
        ratio_u = 1.0 - ratio_s

    assert 0.0 <= ratio_s <= 1.0
    assert 0.0 <= ratio_u <= 1.0
    assert 0.0 <= ratio_s + ratio_u <= 1.0

    targets_multihot = get_multihots(dataset, n_classes, target_type)

    n_samples = targets_multihot.shape[0]
    n_classes = targets_multihot.shape[1]

    if ratio_s == 1.0:
        return torch.arange(n_samples), torch.as_tensor([])

    # get all dataset targets and compute original class distribution metrics
    all_targets_idx = list(range(n_samples))

    # expected occurrence and tolerance
    total_occur = torch.sum(targets_multihot, dim=0)
    s_expected_occur = torch.ceil(total_occur * torch.scalar_tensor(ratio_s)).int()
    u_expected_occur = torch.ceil(total_occur * torch.scalar_tensor(ratio_u)).int()

    # loop through the dataset and constitute the two subset.
    remaining_sample = list(zip(targets_multihot, all_targets_idx))
    if shuffle:
        random.shuffle(remaining_sample)
    indexes_s, remaining_sample = _fill_subset(
        remaining_sample, s_expected_occur, n_classes, verbose
    )

    # For the unsupervised subset, if automatic set, then it is the remaining samples
    if ratio_u + ratio_s == 1.0:
        indexes_u = torch.as_tensor([idx for _target, idx in remaining_sample])
    else:
        indexes_u, _ = _fill_subset(
            remaining_sample, u_expected_occur, n_classes, verbose
        )

    return indexes_s, indexes_u


def get_multihots(
    dataset: SizedDataset, n_classes: int, target_type: str = "indexes"
) -> Tensor:
    """
    :param dataset: TODO
    :param n_classes: TODO
    :param target_type: 'indexes' or 'multihot'
    """
    if not hasattr(dataset, "get_target"):
        raise RuntimeError(
            f'Dataset "{type(dataset)}" must have a method "get_target(index)" for getting multihots targets.'
        )

    n_samples = len(dataset)

    targets = [torch.as_tensor(dataset.get_target(i)) for i in range(n_samples)]  # type: ignore
    if target_type == "indexes":
        multihots = torch.zeros(n_samples, n_classes, dtype=torch.long)
        for i, target in enumerate(targets):
            multihots[i, target] = 1
    elif target_type == "multihot":
        multihots = torch.stack(targets)
    else:
        raise RuntimeError(f'Unknown target_type "{target_type}".')

    return multihots


def get_indexes_per_class(
    dataset: SizedDataset,
    n_classes: int,
    target_type: str = "indexes",
    indexes: Union[Iterable[int], Tensor, None] = None,
) -> list[list[int]]:
    """
    :param dataset: TODO
    :param n_classes: TODO
    :param target_type: 'indexes' or 'multihot'
    :param indexes: TODO
    """
    if indexes is None:
        indexes = list(range(len(dataset)))

    if not hasattr(dataset, "get_target"):
        raise RuntimeError(
            f'Dataset "{type(dataset)}" must have a method "get_target(index)" for getting indexes per class.'
        )

    if isinstance(indexes, Tensor):
        if indexes.is_floating_point() or indexes.ndim != 1:
            raise ValueError(
                f"Invalid argument indexes. (expected a IntTensor with 1 dim but found {indexes.is_floating_point()=} and {indexes.ndim=})"
            )

    indexes_per_class: list = [[] for _ in range(n_classes)]
    targets = [torch.as_tensor(dataset.get_target(i)) for i in indexes]  # type: ignore

    for i, target in zip(indexes, targets):
        if target_type == "indexes":
            target_indexes = target
        elif target_type == "multihot":
            target_indexes = torch.arange(0, len(target))
            target_indexes = target_indexes[target]
        else:
            raise RuntimeError(f'Unknown target_type "{target_type}".')

        for cls_idx in target_indexes:
            indexes_per_class[cls_idx].append(i)

    return indexes_per_class


def get_indexes_per_class_from_multihots(targets: Tensor) -> List[List[int]]:
    n_classes = targets.shape[1]
    return [torch.where(targets[:, i].eq(1.0))[0].tolist() for i in range(n_classes)]


def get_multihots_from_indexes_per_class(
    indexes_per_class: Sequence[Sequence[int]],
) -> Tensor:
    max_idx = -1
    for indexes in indexes_per_class:
        max_idx = max(max_idx, max(indexes))

    n_classes = len(indexes_per_class)
    targets = torch.full((max_idx + 1, n_classes), fill_value=False, dtype=torch.bool)
    for idx_class, indexes in enumerate(indexes_per_class):
        for idx in indexes:
            if targets[idx, idx_class]:
                raise RuntimeError(
                    f'Found duplicate for index "{idx}" and class index "{idx_class}".'
                )
            targets[idx, idx_class] = True
    return targets


def flat_indexes_per_class(indexes_per_class: Sequence[Sequence[int]]) -> List[int]:
    all_indexes = []
    for indexes in indexes_per_class:
        all_indexes += indexes
    return torch.unique(torch.as_tensor(all_indexes, dtype=torch.long)).tolist()


def _fill_subset(
    remaining_samples: List[Tuple[Tensor, int]],
    expected: Tensor,
    n_classes: int,
    verbose: bool = True,
) -> Tuple[Tensor, List[Tuple[Tensor, int]]]:
    subset_occur = torch.zeros(n_classes)
    subset_indexes = []
    total_expected = expected.sum().item()

    if verbose:
        logging.info("Splitting the dataset...")
        progress = tqdm.tqdm(total=total_expected)
    else:
        progress = None

    for class_idx in range(n_classes):
        idx = 0
        while (
            idx < len(remaining_samples)
            and subset_occur[class_idx] < expected[class_idx]
        ):
            if remaining_samples[idx][0][class_idx] == 1:
                target, target_idx = remaining_samples.pop(idx)
                subset_occur += target
                subset_indexes.append(target_idx)
                if progress is not None:
                    progress.update(int(target.sum().item()))

            idx += 1

    if progress is not None:
        progress.close()

    return torch.as_tensor(subset_indexes), remaining_samples


def balanced_split_v2(
    indexes_per_class: Sequence[Sequence[int]],
    ratios: Sequence[float],
    verbose: bool = False,
) -> List[List[List[int]]]:
    targets = get_multihots_from_indexes_per_class(indexes_per_class)
    indexes_per_class = [torch.as_tensor(indexes) for indexes in indexes_per_class]  # type: ignore

    n_classes = len(indexes_per_class)
    n_splits = len(ratios)
    n_elements = targets.shape[0]

    if verbose:
        print(
            f"Info: n_classes={n_classes}, n_splits={n_splits}, n_elements={n_elements}"
        )

    n_expected_per_splits = torch.as_tensor(
        [
            [round(len(indexes) * ratio) for indexes in indexes_per_class]
            for ratio in ratios
        ]
    )

    splits = [[[] for _ in range(n_classes)] for _ in range(n_splits)]

    taken = torch.full((n_elements,), False, dtype=torch.bool)
    n_taken = 0

    while n_taken < n_elements:
        n_by_splits = torch.as_tensor(
            [[len(indexes) for indexes in split] for split in splits]
        )
        n_missing_per_splits = n_expected_per_splits - n_by_splits

        if verbose:
            n_missing_total = n_missing_per_splits.sum().item()
            print(
                f"[{n_taken}/{n_elements}] taken. Missing: {n_missing_total}. ",
                end="\r",
            )

        # Search the max missing elem
        idx_ratio_prior, idx_class_prior = torch.where(
            n_missing_per_splits.eq(n_missing_per_splits.max())
        )
        random_prior = torch.randint(len(idx_ratio_prior), (1,))
        idx_ratio_prior = idx_ratio_prior[random_prior].item()
        idx_class_prior = idx_class_prior[random_prior].item()

        # Search if one elem of this class is available
        indexes = indexes_per_class[idx_class_prior]  # type: ignore
        taken_class = torch.where(taken[indexes].eq(False))[0]

        if len(taken_class) > 0:
            random_in_class = torch.randint(len(taken_class), (1,))
            found_idx = indexes[taken_class[random_in_class]]

            # Search the other classes of this elem 'found_idx'
            classes_of_found_idx = torch.where(targets[found_idx])[0].tolist()

            for idx_class in classes_of_found_idx:
                splits[idx_ratio_prior][idx_class].append(found_idx)  # type: ignore
            taken[found_idx] = True
            n_taken += 1

        else:
            # If no elem of the class search is available, ignore this class now
            n_expected_per_splits[idx_ratio_prior][idx_class_prior] = n_by_splits[  # type: ignore
                idx_ratio_prior
            ][
                idx_class_prior
            ]

    if verbose:
        print(f"[{n_taken}/{n_elements}] taken.", end="\n")
    return splits
