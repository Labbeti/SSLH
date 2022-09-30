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


class UDA(LightningModule):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        activation: Module = Softmax(dim=-1),
        criterion_s: Module = NLLLossVecTargets(reduction="none"),
        criterion_u: Module = NLLLossVecTargets(reduction="none"),
        lambda_u: float = 1.0,
        threshold: float = 0.8,
        temperature: float = 0.4,
        train_metrics: Optional[Dict[str, Module]] = None,
        val_metrics: Optional[Dict[str, Module]] = None,
        log_on_epoch: bool = True,
    ):
        """
        Unsupervised Data Augmentation (UDA) LightningModule.

        :param model: The PyTorch Module to train.
                The forward() must return logits for classify the data.
        :param optimizer: The PyTorch optimizer to use.
        :param activation: The activation function of the model.
                (default: Softmax(dim=-1))
        :param criterion_s: The loss component 'L_s' of RMM.
                (default: CrossEntropyWithVectors())
        :param criterion_u: The loss component 'L_u' of RMM.
                (default: CrossEntropyWithVectors())
        :param lambda_u: The coefficient of the 'L_u' component.
                (default: 1.0)
        :param threshold: The confidence threshold 'tau' used for the mask of the 'L_u' component.
                (default: 0.8)
        :param temperature: The temperature 'T' used for post-process the pseudo-label
                (default: 0.4)
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
        self.criterion_s = criterion_s
        self.criterion_u = criterion_u
        self.lambda_u = lambda_u
        self.threshold = threshold
        self.temperature = temperature

        self.metric_dict_train_s = ForwardDictAffix(
            train_metrics, prefix="train/", suffix="_s"
        )
        self.metric_dict_train_u_pseudo = ForwardDictAffix(
            train_metrics, prefix="train/", suffix="_u"
        )
        self.metric_dict_val = ForwardDictAffix(val_metrics, prefix="val/")
        self.metric_dict_test = ForwardDictAffix(val_metrics, prefix="test/")

        self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)

        self.save_hyperparameters(
            {
                "experiment": self.__class__.__name__,
                "model": model.__class__.__name__,
                "activation": activation.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "criterion_s": criterion_s.__class__.__name__,
                "criterion_u": criterion_u.__class__.__name__,
                "lambda_u": lambda_u,
                "threshold": threshold,
                "temperature": temperature,
            }
        )

    def training_step(
        self,
        batch: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]],
        batch_idx: int,
    ):
        (xs, ys), (xu, xu_strong) = batch

        # Compute pseudo-labels 'yu' and mask
        yu, mask = self.guess_label_and_mask(xu)

        # Compute predictions on xs and xu
        probs_xs = self.activation(self.model(xs))
        probs_xu_strong = self.activation(self.model(xu_strong))

        # Criterion (loss_s of shape bsize_s, loss_u of shape bsize_u)
        loss_s = self.criterion_s(probs_xs, ys)
        loss_u = self.criterion_u(probs_xu_strong, yu)

        loss_s = torch.mean(loss_s)
        loss_u = torch.mean(loss_u * mask)

        loss = loss_s + self.lambda_u * loss_u

        with torch.no_grad():
            scores = {
                "loss": loss,
                "loss_s": loss_s,
                "loss_u": loss_u,
                "mask": mask.mean(),
            }
            scores = {f"train/{k}": v.cpu() for k, v in scores.items()}
            self.log_dict(scores, **self.log_params)

            scores_s = self.metric_dict_train_s(probs_xs, ys)
            self.log_dict(scores_s, **self.log_params)

            probs_xu = self.activation(self.model(xu))
            scores_u = self.metric_dict_train_u_pseudo(probs_xu, yu)
            self.log_dict(scores_u, **self.log_params)

        return loss

    def guess_label_and_mask(self, xu: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            logits_xu = self.model(xu)
            probs_xu = self.activation(logits_xu)
            probabilities_max, _ = probs_xu.max(dim=-1)
            mask = probabilities_max.ge(self.threshold).to(probs_xu.dtype)
            yu = torch.softmax(logits_xu / self.temperature, dim=-1)
            return yu, mask

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        xs, ys = batch
        probs_xs = self(xs)
        self.log_dict(self.metric_dict_val(probs_xs, ys), on_epoch=True, on_step=False)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        xs, ys = batch
        probs_xs = self(xs)
        self.log_dict(self.metric_dict_test(probs_xs, ys), on_epoch=True, on_step=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.model(x))

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer
