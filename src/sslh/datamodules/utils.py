#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Iterable, List, Tuple, Union

logger = logging.getLogger(__name__)


def guess_folds(
    train_folds: Union[Iterable[int], int, None],
    val_folds: Union[Iterable[int], int, None],
    folds: Iterable[int],
    verbose: int = 0,
) -> Tuple[List[int], List[int]]:
    """
    Use the train_folds and folds to guess val_folds OR use the val_folds and folds to guess train_folds.

    :param train_folds: The training folds.
    :param val_folds: The validation folds.
    :param folds: The list of folds.
    :return: A tuple of folds (training folds, validation folds).
    """
    if verbose >= 2:
        logger.debug(f"Guessing folds with {train_folds=} and {val_folds=}. ({folds=})")

    if isinstance(train_folds, int):
        train_folds = [train_folds]
    if isinstance(val_folds, int):
        val_folds = [val_folds]
    folds = set(folds)

    if train_folds is None and val_folds is None:
        if verbose >= 1:
            logger.info(
                f"Found {train_folds=} and {val_folds=}, use fold {len(folds)} as validation the others as train by default."
            )
        val_folds = [len(folds)]
        train_folds = folds.difference(val_folds)

    elif train_folds is None:
        assert isinstance(val_folds, Iterable)
        train_folds = folds.difference(val_folds)

    elif val_folds is None:
        assert isinstance(train_folds, Iterable)
        val_folds = folds.difference(train_folds)

    train_folds = list(train_folds)
    val_folds = list(val_folds)

    if verbose >= 2:
        logger.debug(f"Found folds {train_folds=} and {val_folds=}.")

    return train_folds, val_folds


def guess_num_workers_su(
    n_workers: int, size_s: int, size_u: int, verbose: int
) -> Tuple[int, int]:
    if n_workers >= 2:
        num_workers_s = min(
            max(
                round(size_s / (size_s + size_u) * n_workers),
                1,
            ),
            n_workers - 1,
        )
        num_workers_u = n_workers - num_workers_s
        if verbose >= 2:
            logger.debug(
                f"Use {num_workers_s=} and {num_workers_u=}. (with {size_s=} and {size_u=})"
            )

    else:
        num_workers_s = 0
        num_workers_u = 0
        if verbose >= 0:
            logger.warning(
                f"Not enough workers for train dataloaders. (found {n_workers=} < 2)"
            )

    return num_workers_s, num_workers_u
