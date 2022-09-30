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
