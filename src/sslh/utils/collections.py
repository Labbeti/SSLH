#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterable, Mapping


def flat_dict_of_dict(
    nested_dic: Mapping[str, Any],
    sep: str = ".",
    flat_iterables: bool = False,
) -> Dict[str, Any]:
    """Flat a nested dictionary.

    Example
    ----------
    ```
    >>> dic = {
        "a": 1,
        "b": {
            "a": 2,
            "b": 10,
        },
    }
    >>> flat_dict(dic)
    ... {"a": 1, "b.a": 2, "b.b": 10}
    ```
    """
    output = {}
    for k, v in nested_dic.items():
        if isinstance(v, Mapping) and all(isinstance(kv, str) for kv in v.keys()):
            v = flat_dict_of_dict(v, sep, flat_iterables)
            output |= {f"{k}{sep}{kv}": vv for kv, vv in v.items()}
        elif flat_iterables and isinstance(v, Iterable) and not isinstance(v, str):
            output |= {
                f"{k}{sep}{i}": flat_dict_of_dict(vv, sep, flat_iterables)
                for i, vv in enumerate(v)
            }
        else:
            output[k] = v
    return output
