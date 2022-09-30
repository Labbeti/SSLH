#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple

import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer

from sslh.nn.loss import NLLLossVecTargets
from sslh.nn.utils import ForwardDictAffix


class Supervised(LightningModule):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        activation: Module = Softmax(dim=-1),
        criterion: Module = NLLLossVecTargets(),
        train_metrics: Optional[Dict[str, Module]] = None,
        val_metrics: Optional[Dict[str, Module]] = None,
        log_on_epoch: bool = True,
    ):
        """
        Supervised training LightningModule.

        :param model: The PyTorch Module to train.
                The forward() must return logits for classify the data.
        :param optimizer: The PyTorch optimizer to use.
        :param activation: The activation function of the model.
                (default: Softmax(dim=-1))
        :param criterion: The criterion used for compute loss.
                (default: CrossEntropyWithVectors())
        :param train_metrics: An optional dictionary of metrics modules for training.
                (default: None)
        :param val_metrics: An optional dictionary of metrics modules for validation.
                (default: None)
        :param log_on_epoch: If True, log only the epoch means of each train metric score.
                (default: True)
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.activation = activation
        self.criterion = criterion
        self.metric_dict_train = ForwardDictAffix(train_metrics, prefix="train/")
        self.metric_dict_val = ForwardDictAffix(val_metrics, prefix="val/")
        self.metric_dict_test = ForwardDictAffix(val_metrics, prefix="test/")

        self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)

        self.save_hyperparameters(
            {
                "experiment": self.__class__.__name__,
                "model": model.__class__.__name__,
                "activation": activation.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "criterion": criterion.__class__.__name__,
            }
        )

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        xs, ys = batch

        probs_xs = self.activation(self.model(xs))
        loss = self.criterion(probs_xs, ys)

        with torch.no_grad():
            scores = {"loss": loss}
            scores = {f"train/{k}": v.cpu() for k, v in scores.items()}
            self.log_dict(scores, **self.log_params)

            scores = self.metric_dict_train(probs_xs, ys)
            self.log_dict(scores, **self.log_params)

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        xs, ys = batch
        probs_xs = self(xs)
        self.log_dict(self.metric_dict_val(probs_xs, ys), on_epoch=True, on_step=False)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        xs, ys = batch
        probs_xs = self(xs)
        self.log_dict(self.metric_dict_test(probs_xs, ys), on_epoch=True, on_step=False)

    def forward(self, x: Tensor) -> Tensor:
        probs_x = self.activation(self.model(x))
        return probs_x

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer
