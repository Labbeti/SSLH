#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

from typing import Any, Optional

from torch import nn


class EMA(nn.Module):
    BUFFERS_MODES = ("ema", "set", "none")

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.999,
        copy_model: bool = False,
        buffers_mode: str = "ema",
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
        :param buffers_mode: The mode to synchronize buffers. defaults to "ema".
            "ema" => compute ema on floating point buffers.
            "set" => set ema buffers to student target buffers.
            "none" => leave buffers as is.
        """
        if buffers_mode not in self.BUFFERS_MODES:
            raise ValueError(
                f"Invalid argument {buffers_mode=}. Must be one of {self.BUFFERS_MODES}."
            )

        if copy_model:
            model = copy.deepcopy(model)

        super().__init__()
        self.model = model
        self.alpha = alpha
        self.buffers_mode = buffers_mode

    def ema_update(self, student: nn.Module, step: Optional[int]) -> None:
        alpha = self.get_cur_alpha(step)
        for param, ema_param in zip(student.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1.0 - alpha)

        if self.buffers_mode == "ema":
            for buffer, ema_buffer in zip(student.buffers(), self.model.buffers()):
                assert buffer.is_floating_point() == ema_buffer.is_floating_point()
                if buffer.is_floating_point():
                    ema_buffer.data.mul_(alpha).add_(buffer.data, alpha=1.0 - alpha)

        elif self.buffers_mode == "set":
            for buffer, ema_buffer in zip(student.buffers(), self.model.buffers()):
                ema_buffer[:] = buffer

        elif self.buffers_mode == "none":
            pass
        else:
            raise ValueError(
                f"Invalid parameter {self.buffers_mode=}. Must be one of {self.BUFFERS_MODES}."
            )

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def get_cur_alpha(self, step: Optional[int]) -> float:
        alpha = self.alpha
        if step is not None:
            # Use the true average until the exponential average is more correct
            alpha = min(1 - 1 / (step + 1), alpha)
        return alpha

    def extra_repr(self) -> str:
        return f"decay={self.alpha}"
