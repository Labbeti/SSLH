#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Union

import torch

from pytorch_lightning import Callback, LightningModule, Trainer
from torch.utils.data.dataloader import DataLoader

from sslh.nn.utils import ForwardDictAffix


class ValidationCallback(Callback):
    def __init__(
        self,
        metric_dict_val: ForwardDictAffix,
        stack_metric: bool = False,
        val_check_interval: Union[float, int] = 1.0,
        **forward_kwargs,
    ):
        super().__init__()
        self.metric_dict_val = metric_dict_val
        self.stack_metric = stack_metric
        self.val_check_interval = val_check_interval
        self.forward_kwargs = forward_kwargs

        self.log_params = dict(on_epoch=False, on_step=True, prog_bar=True)
        if isinstance(val_check_interval, int) and val_check_interval >= 0:
            self._val_step_interval = val_check_interval
        elif isinstance(val_check_interval, float) and val_check_interval >= 0.0:
            self._val_step_interval = None
        else:
            raise RuntimeError(
                'Parameter "val_check_interval" must be a positive integer or float.'
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        del batch
        if self._val_step_interval is None:
            dataloader = trainer.train_dataloader
            self._val_step_interval = round(len(dataloader) * self.val_check_interval)

        global_step = trainer.global_step
        if (
            self._val_step_interval != 0
            and global_step % self._val_step_interval == self._val_step_interval - 1
        ):
            self.run(trainer, pl_module)

    def run(self, trainer: Trainer, pl_module: LightningModule):
        with torch.no_grad():
            dataloaders = trainer.val_dataloaders

            if len(dataloaders) == 0:
                pass
            elif len(dataloaders) == 1:
                dataloader = dataloaders[0]
                if not self.stack_metric:
                    self.run_validation(dataloader, pl_module)
                else:
                    self.run_validation_stack(dataloader, pl_module)
            else:
                raise RuntimeError(
                    f"Unsupported multiple dataloaders for {self.__class__.__name__}."
                )

    def run_validation(self, dataloader: DataLoader, pl_module: LightningModule):
        concat_scores = {}
        for batch in dataloader:
            xs, ys = batch
            xs = xs.to(dtype=pl_module.dtype, device=pl_module.device)
            ys = ys.to(dtype=pl_module.dtype, device=pl_module.device)

            probs_xs = pl_module(xs, **self.forward_kwargs)
            scores = self.metric_dict_val(probs_xs, ys)

            for name, score in scores.items():
                if name not in concat_scores.keys():
                    concat_scores[name] = score
                else:
                    concat_scores[name] += score

        for name in concat_scores.keys():
            concat_scores[name] /= len(dataloader)
        pl_module.log_dict(concat_scores, **self.log_params)

    def run_validation_stack(self, dataloader: DataLoader, pl_module: LightningModule):
        list_pred = []
        list_ys = []

        for batch in dataloader:
            xs, ys = batch
            xs = xs.to(dtype=pl_module.dtype, device=pl_module.device)
            ys = ys.to(dtype=pl_module.dtype, device=pl_module.device)

            pred = pl_module(xs)

            list_pred.append(pred)
            list_ys.append(ys)

        list_pred = torch.vstack(list_pred)
        list_ys = torch.vstack(list_ys)

        scores = self.metric_dict_val(list_pred, list_ys)
        pl_module.log_dict(scores, **self.log_params)
