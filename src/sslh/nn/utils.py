#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect

from typing import Any, Callable, Dict, Mapping, Optional

from torch import nn


class ForwardDict(nn.ModuleDict):
    def __init__(self, modules: Optional[Mapping[str, nn.Module]], **kwargs) -> None:
        """
        Compute output of each nn.Module stored when forward() is called.
        Subclass of Dict[str, nn.Module] and nn.Module.
        """
        if modules is None:
            modules = {}
        elif not isinstance(modules, dict):
            modules = dict(modules)
        modules |= kwargs
        super().__init__(modules)

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        return {name: module(*args, **kwargs) for name, module in self.items()}

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.items())))


class ForwardDictAffix(ForwardDict):
    """
    Compute score of each callable object stored when forward() is applied.
    Subclass of Dict[str, nn.Module] and nn.Module.
    """

    def __init__(
        self,
        modules: Optional[Mapping[str, nn.Module]],
        prefix: str = "",
        suffix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(modules, **kwargs)
        self.prefix = prefix
        self.suffix = suffix

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        return {
            self.prefix + name + self.suffix: output
            for name, output in super().forward(*args, **kwargs).items()
        }

    def to_dict(self, with_affixes: bool = True) -> Dict[str, nn.Module]:
        if with_affixes:
            return {
                self.prefix + name + self.suffix: module
                for name, module in self.items()
            }
        else:
            return dict(self)


def get_n_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """
    Return the number of parameters in a module.

    :param model: Pytorch Module to check.
    :param only_trainable: If True, count only parameters that requires gradient. (default: True)
    :returns: The number of parameters.
    """
    params = (
        param
        for param in model.parameters()
        if not only_trainable or param.requires_grad
    )
    return sum(param.numel() for param in params)


class Lambda(nn.Module):
    def __init__(self, fn: Callable) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs) -> Any:
        return self.fn(*args, **kwargs)

    def extra_repr(self) -> str:
        if inspect.isfunction(self.fn):
            return self.fn.__name__
        elif inspect.ismethod(self.fn):
            return self.fn.__qualname__
        else:
            return self.fn.__class__.__name__
