#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple

import torch

from torch import nn, Tensor
from torch.optim.optimizer import Optimizer

from sslh.nn.loss import CrossEntropyLossVecTargets
from sslh.pl_modules.mixup.mixup_nolabelmix import MixupNoLabelMix


class Mixup(MixupNoLabelMix):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        activation: nn.Module = nn.Softmax(dim=-1),
        criterion: nn.Module = CrossEntropyLossVecTargets(),
        alpha: float = 0.4,
        train_metrics: Optional[Dict[str, nn.Module]] = None,
        val_metrics: Optional[Dict[str, nn.Module]] = None,
        log_on_epoch: bool = True,
    ) -> None:
        """
        Mixup with mix labels (MUM) LightningModule.

        :param model: The PyTorch Module to train.
                The forward() must return logits to classify the data.
        :param optimizer: The PyTorch optimizer to use.
        :param activation: The activation function of the model.
                (default: Softmax(dim=-1))
        :param criterion: The criterion used for compute loss.
                (default: CrossEntropyWithVectors())
        :param alpha: The mixup alpha parameter. A higher value means a stronger mix between labeled and unlabeled data.
                (default: 0.75)
        :param train_metrics: An optional dictionary of metrics modules for training.
                (default: None)
        :param val_metrics: An optional dictionary of metrics modules for validation.
                (default: None)
        :param log_on_epoch: If True, log only the epoch means of each train metric score.
                (default: True)
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            activation=activation,
            criterion=criterion,
            alpha=alpha,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            log_on_epoch=log_on_epoch,
        )

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        xs, ys = batch

        with torch.no_grad():
            indexes = torch.randperm(xs.shape[0])
            xs_shuffle = xs[indexes]
            ys_shuffle = ys[indexes]

            xs_mix, ys_mix = self.mixup(xs, xs_shuffle, ys, ys_shuffle)

        logits_xs_mix = self.model(xs_mix)
        loss = self.criterion(logits_xs_mix, ys_mix)

        with torch.no_grad():
            scores = {"loss": loss}
            scores = {f"train/{k}": v.cpu() for k, v in scores.items()}
            self.log_dict(scores, **self.log_params)

            scores = self.metric_dict_train(self.activation(logits_xs_mix), ys)
            self.log_dict(scores, **self.log_params)

        return loss
