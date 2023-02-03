#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from abc import ABC
from typing import Any

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LRSchedulerCallback(Callback, LambdaLR, ABC):
    def __init__(
        self,
        optimizer: Optimizer,
        on_epoch: bool = True,
    ) -> None:
        # Note: self.n_epochs must be defined before super().__init__ call !
        self.n_steps = 1
        self.on_epoch = on_epoch
        super().__init__(optimizer=optimizer, lr_lambda=self._lr_lambda_torch)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
    ) -> None:
        if self.on_epoch:
            self.n_steps = trainer.max_epochs  # type: ignore
            self.step()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not self.on_epoch:
            self.n_steps = trainer.num_training_batches * trainer.max_epochs  # type: ignore
            self.step()

    def _lr_lambda_torch(self, step: int) -> float:
        return self.lr_lambda(step, self.n_steps)

    def lr_lambda(self, step: int, n_steps: int) -> float:
        raise NotImplementedError("Abstract method")


class CosineScheduler(LRSchedulerCallback):
    """
    Use learning rate using the following eq:

    >>> 'cos(7 / 16 * pi * step / n_steps)'
    """

    def lr_lambda(self, step: int, n_steps: int) -> float:
        return math.cos(7.0 / 16.0 * math.pi * min(step / n_steps, 1.0))


class SoftCosineScheduler(LRSchedulerCallback):
    """
    Use learning rate using the following eq:

    >>> '0.5 * (1 + cos((step - 1) * pi / n_steps))'
    """

    def lr_lambda(self, step: int, n_steps: int) -> float:
        return 0.5 * (1.0 + math.cos(math.pi * min(step / n_steps, 1.0)))
