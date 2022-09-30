#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from typing import Any, Optional

from pytorch_lightning.callbacks import Callback


class WarmUpCallback(Callback):
    WARMUP_RULES = ("constant", "linear_increase", "exp_increase")

    def __init__(
        self,
        target_value: float,
        warmup_len: int,
        warmup_rule: str = "linear_increase",
        on_epoch: bool = False,
        target_obj: Optional[object] = None,
        target_attr: Optional[str] = None,
        start_value: float = 0.0,
    ) -> None:
        """
        Note: exp_increase warmup : https://arxiv.org/pdf/1803.05984.pdf and https://arxiv.org/pdf/1610.02242.pdf
        """
        if warmup_rule not in self.WARMUP_RULES:
            raise ValueError(
                f"Invalid argument {warmup_rule=}. Expected one of {self.WARMUP_RULES}."
            )
        if (target_obj is not None and target_attr is None) or (
            target_obj is None and target_attr is not None
        ):
            raise ValueError(
                f"Invalid combinaison of arguments {target_obj=} and {target_attr=}. (expected (None, None) or (object, str))"
            )

        super().__init__()
        self.target_value = target_value
        self.warmup_rule = warmup_rule
        self.warmup_len = warmup_len
        self.on_epoch = on_epoch
        self.target_obj = target_obj
        self.target_attr = target_attr
        self.start_value = start_value

        self._cur_warmup_step = 0

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not self.on_epoch:
            self.update()

    def on_train_epoch_end(self, trainer, pl_module, outputs: Any) -> None:
        if self.on_epoch:
            self.update()

    def update(self) -> None:
        if self.target_obj is not None and self.target_attr is not None:
            cur_value = self.get_cur_value()
            self.target_obj.__setattr__(self.target_attr, cur_value)  # type: ignore
        self._cur_warmup_step += 1

    def get_cur_value(self) -> float:
        if self.warmup_rule == "constant":
            coef = 1.0
        elif self.warmup_rule == "linear_increase":
            coef = self._cur_warmup_step / self.warmup_len
        elif self.warmup_rule == "exp_increase":
            coef = math.exp(
                -5.0 * (1.0 - min(self._cur_warmup_step / self.warmup_len, 1.0)) ** 2
            )
        else:
            raise ValueError(
                f"Invalid argument {self.warmup_rule=}. Expected one of {self.WARMUP_RULES}."
            )

        cur_value = coef * (self.target_value - self.start_value) + self.start_value
        return cur_value

    def get_cur_step(self) -> int:
        return self._cur_warmup_step
