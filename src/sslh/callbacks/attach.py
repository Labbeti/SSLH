#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer


class AttachExampleInputArray(Callback):
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._attach(trainer, pl_module)

    def _attach(self, trainer: Trainer, pl_module: LightningModule) -> None:
        datamodule = pl_module.datamodule

        if (
            datamodule is not None
            and hasattr(datamodule, "example_input_array")
            and pl_module.example_input_array is None
        ):
            pl_module.example_input_array = datamodule.example_input_array
