#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import subprocess

from typing import Optional

import numpy as np
import torch


def reset_seed(seed: Optional[int]) -> None:
    """Reset the seed of following packages for reproductibility :
    - random
    - numpy
    - torch
    - torch.cuda

    Also set deterministic behaviour for cudnn backend.

    :param seed: The seed to set.
    """
    if seed is not None and not isinstance(seed, int):
        raise TypeError(
            f"Invalid argument type {type(seed)=}. (expected NoneType or int)"
        )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):  # type: ignore
            torch.backends.cudnn.benchmark = False  # type: ignore
            torch.backends.cudnn.deterministic = True  # type: ignore
        else:
            raise RuntimeError(
                "Cannot make deterministic behaviour for current torch backend (cannot find submodule torch.backends.cudnn)."
            )


def get_current_git_hash() -> str:
    """
    Return the current git hash in the current directory.

    :returns: The git hash. If an error occurs, returns 'UNKNOWN'.
    """
    try:
        git_hash = subprocess.check_output(["git", "describe", "--always"])
        git_hash = git_hash.decode("UTF-8").replace("\n", "")
        return git_hash
    except (subprocess.CalledProcessError, PermissionError):
        return "UNKNOWN"
