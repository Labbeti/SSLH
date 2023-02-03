#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp

from logging import FileHandler
from typing import Optional


class CustomFileHandler(FileHandler):
    """FileHandler that build intermediate directories.
    Used for export hydra logs to a file contained in a folder that does not exists yet at the start of the program.
    """

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        encoding: Optional[str] = None,
        delay: bool = True,
    ) -> None:
        parent_dpath = osp.dirname(filename)
        if parent_dpath != "":
            try:
                os.makedirs(parent_dpath, exist_ok=True)
            except PermissionError:
                pass
        super().__init__(filename, mode, encoding, delay)
