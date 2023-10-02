#!/usr/bin/env python
# -*- coding: utf-8 -*-


def print_usage() -> None:
    print(
        "Usage:\n"
        "- python -m sslh.deep_co_training [ARGS...]\n"
        "- python -m sslh.fixmatch [ARGS...]\n"
        "- python -m sslh.mean_teacher [ARGS...]\n"
        "- python -m sslh.mixmatch [ARGS...]\n"
        "- python -m sslh.mixup [ARGS...]\n"
        "- python -m sslh.pseudo_labeling [ARGS...]\n"
        "- python -m sslh.remixmatch [ARGS...]\n"
        "- python -m sslh.supervised [ARGS...]\n"
        "- python -m sslh.uda [ARGS...]\n"
        "- python -m sslh.version"
    )


if __name__ == "__main__":
    print_usage()
