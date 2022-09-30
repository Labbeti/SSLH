#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

from typing import Optional

from torch import nn


class EMA(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.999,
        copy_model: bool = False,
    ) -> None:
        """Compute the exponential moving average (EMA) of a model.

        `model = alpha * model + (1 - alpha) * other_model`

        Usage :
        >>> linear = nn.Linear(100, 10)
        >>> ema = EMA(linear, alpha=0.9)
        >>> # train linear module...
        >>> ema.update(linear)
        >>> # ema.model now contains `linear * 0.1 + old_linear * 0.9`

        :param model: The target model to update.
        :param decay: The exponential decay (also called 'alpha') used to update the model. (default: 0.999)
        :param copy_model: If True, the model passed as input will be copied. (default: False)
        """
        if copy_model:
            model = copy.deepcopy(model)

        super().__init__()
        self.model = model
        self.alpha = alpha

    def ema_update(self, student: nn.Module, step: Optional[int]) -> None:
        alpha = self.get_cur_alpha(step)
        for param, ema_param in zip(student.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_cur_alpha(self, step: Optional[int]) -> float:
        alpha = self.alpha
        if step is not None:
            # Use the true average until the exponential average is more correct
            alpha = min(1 - 1 / (step + 1), alpha)
        return alpha

    def extra_repr(self) -> str:
        return f"decay={self.alpha}"
