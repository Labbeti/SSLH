#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple

import torch

from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer

from sslh.nn.utils import ForwardDictAffix
from sslh.nn.loss import CrossEntropyLossVecTargets
from sslh.transforms.other.mixup import MixupModule


class MixupNoLabelMix(LightningModule):
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
        Mixup (MU) LightningModule.

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
        super().__init__()
        self.model = model
        self.activation = activation
        self.optimizer = optimizer
        self.criterion = criterion
        self.alpha = alpha

        self.metric_dict_train = ForwardDictAffix(train_metrics, prefix="train/")
        self.metric_dict_val = ForwardDictAffix(val_metrics, prefix="val/")
        self.metric_dict_test = ForwardDictAffix(val_metrics, prefix="test/")

        self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)
        self.mixup = MixupModule(alpha=alpha, apply_max=False)

        self.save_hyperparameters(
            {
                "experiment": self.__class__.__name__,
                "model": model.__class__.__name__,
                "activation": activation.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "criterion": criterion.__class__.__name__,
                "alpha": alpha,
            }
        )

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        xs, ys = batch

        with torch.no_grad():
            indexes = torch.randperm(xs.shape[0])
            xs_shuffle = xs[indexes]
            ys_shuffle = ys[indexes]

            xs_mix, _ys_mix = self.mixup(xs, xs_shuffle, ys, ys_shuffle)
            lambda_ = self.mixup.get_last_lambda()

        logits_xs_mix = self.model(xs_mix)
        loss = lambda_ * self.criterion(logits_xs_mix, ys) + (
            1.0 - lambda_
        ) * self.criterion(logits_xs_mix, ys_shuffle)

        with torch.no_grad():
            scores = {"loss": loss}
            scores = {f"train/{k}": v.cpu() for k, v in scores.items()}
            self.log_dict(scores, **self.log_params)

            scores = self.metric_dict_train(self.activation(self.model(xs)), ys)
            self.log_dict(scores, **self.log_params)

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        xs, ys = batch
        probs_xs = self(xs)
        self.log_dict(
            self.metric_dict_val(probs_xs, ys),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        xs, ys = batch
        probs_xs = self(xs)
        self.log_dict(
            self.metric_dict_test(probs_xs, ys),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.model(x))

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer
