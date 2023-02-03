#!/usr/bin/env python
# -*- coding: utf-8 -*-

import platform
import sys

import pytorch_lightning
import torch
import yaml

import sslh


def get_packages_versions() -> dict[str, str]:
    return {
        "sslh": sslh.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os": platform.system(),
        "architecture": platform.architecture()[0],
        "pytorch": torch.__version__,  # type: ignore
        "pytorch_lightning": pytorch_lightning.__version__,
    }


def main_version(*args, **kwargs) -> None:
    """Print some packages versions."""
    versions = get_packages_versions()
    print(yaml.dump(versions, sort_keys=False))


if __name__ == "__main__":
    main_version()
