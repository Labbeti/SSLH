#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from typing import Dict, Optional, Tuple

import torch

from advertorch.attacks import GradientSignAttack
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Optimizer

from sslh.nn.loss import CrossEntropyLossVecTargets
from sslh.nn.utils import ForwardDictAffix


class DeepCoTraining(LightningModule):
    def __init__(
        self,
        model_f: nn.Module,
        model_g: nn.Module,
        optimizer: Optimizer,
        activation: nn.Module = nn.Softmax(dim=-1),
        log_activation: nn.Module = nn.LogSoftmax(dim=-1),
        criterion_s: nn.Module = CrossEntropyLossVecTargets(log_input=True),
        epsilon: float = 0.02,
        lambda_cot: float = 1.0,
        lambda_diff: float = 0.5,
        train_metrics: Optional[Dict[str, nn.Module]] = None,
        val_metrics: Optional[Dict[str, nn.Module]] = None,
        log_on_epoch: bool = True,
    ) -> None:
        """
        Deep Co-Training (DCT) LightningModule.

        :param model_f: The first PyTorch Module to train.
                The forward() must return logits to classify the data.
        :param model_g: The second PyTorch Module to train.
                The forward() must return logits to classify the data.
        :param optimizer: The optimizer used for f and g.
        :param activation: The activation function for the two models.
                (default: Softmax(dim=-1))
        :param log_activation: The log-activation function for the two models.
                (default: LogSoftmax(dim=-1))
        :param criterion_s: The criterion used for loss_sup. Must take log-probabilities as input.
                (default: CrossEntropyWithVectors(log_input=True))
        :param epsilon: The epsilon hparam used for generate adversaries.
                (default: 0.02)
        :param lambda_cot: The lambda_cot coefficient for criterion.
                (default: 1.0)
        :param lambda_diff: The lambda_diff coefficient for criterion.
                (default: 0.5)
        :param train_metrics: An optional dictionary of metrics modules for training.
                (default: None)
        :param val_metrics: An optional dictionary of metrics modules for validation.
                (default: None)
        :param log_on_epoch: If True, log only the epoch means of each train metric score.
                (default: True)
        """
        super().__init__()
        self.model_f = model_f
        self.model_g = model_g
        self.optimizer = optimizer
        self.activation = activation
        self.log_activation = log_activation
        self.criterion_s = criterion_s
        self.epsilon = epsilon
        self.lambda_cot = lambda_cot
        self.lambda_diff = lambda_diff

        self.metric_dict_train_f_s = ForwardDictAffix(
            train_metrics, prefix="train/", suffix="_fs"
        )
        self.metric_dict_train_g_s = ForwardDictAffix(
            train_metrics, prefix="train/", suffix="_gs"
        )
        self.metric_dict_val_f = ForwardDictAffix(val_metrics, prefix="val/")
        self.metric_dict_val_g = ForwardDictAffix(
            val_metrics, prefix="val/", suffix="_g"
        )
        self.metric_dict_test_f = ForwardDictAffix(
            val_metrics, prefix="test/", suffix="_f"
        )
        self.metric_dict_test_g = ForwardDictAffix(
            val_metrics, prefix="test/", suffix="_g"
        )

        self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)

        gsa_params = dict(
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=epsilon,
            clip_min=-math.inf,
            clip_max=math.inf,
            targeted=False,
        )
        self.adv_generator_f = GradientSignAttack(model_f, **gsa_params)
        self.adv_generator_g = GradientSignAttack(model_g, **gsa_params)

        self.save_hyperparameters(
            {
                "experiment": self.__class__.__name__,
                "model": model_f.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "activation": activation.__class__.__name__,
                "log_activation": log_activation.__class__.__name__,
                "criterion_s": criterion_s.__class__.__name__,
                "epsilon": epsilon,
                "lambda_cot": lambda_cot,
                "lambda_diff": lambda_diff,
            }
        )

    def training_step(
        self,
        batch: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor],
        batch_idx: int,
    ) -> Tensor:
        (xs1, ys1), (xs2, ys2), xu = batch

        logits_f_xs1 = self.model_f(xs1)
        logits_g_xs2 = self.model_g(xs2)
        logits_f_xu = self.model_f(xu)
        logits_g_xu = self.model_g(xu)

        with torch.no_grad():
            probs_idx_f_xu = logits_f_xu.argmax(dim=-1)
            probs_idx_g_xu = logits_g_xu.argmax(dim=-1)

            ys1_indices = ys1.argmax(dim=-1)
            ys2_indices = ys2.argmax(dim=-1)

        adv_f_xs1 = self.adv_generator_f.perturb(xs1, ys1_indices)
        adv_f_xu = self.adv_generator_f.perturb(xu, probs_idx_f_xu)

        adv_g_xs2 = self.adv_generator_g.perturb(xs2, ys2_indices)
        adv_g_xu = self.adv_generator_g.perturb(xu, probs_idx_g_xu)

        # Note: logits of model 'f' for the adversarial example of model 'g' of batch 'xs2'
        logits_f_adv_g_xs2 = self.model_f(adv_g_xs2)
        logits_g_adv_f_xs1 = self.model_g(adv_f_xs1)

        logits_f_adv_g_xu = self.model_f(adv_g_xu)
        logits_g_adv_f_xu = self.model_g(adv_f_xu)

        # Compute losses
        loss_sup = self.loss_sup(logits_f_xs1, logits_g_xs2, ys1, ys2)
        loss_cot = self.loss_cot(logits_f_xu, logits_g_xu)
        loss_diff = self.loss_diff(
            logits_f_xs1,
            logits_g_xs2,
            logits_f_adv_g_xs2,
            logits_g_adv_f_xs1,
            logits_f_xu,
            logits_g_xu,
            logits_f_adv_g_xu,
            logits_g_adv_f_xu,
        )

        loss = loss_sup + self.lambda_cot * loss_cot + self.lambda_diff * loss_diff

        with torch.no_grad():
            # Compute metrics
            scores = {
                "loss": loss,
                "loss_sup": loss_sup,
                "loss_cot": loss_cot,
                "loss_diff": loss_diff,
            }
            scores = {f"train/{k}": v.cpu() for k, v in scores.items()}
            self.log_dict(scores, **self.log_params)

            scores_f = self.metric_dict_train_f_s(self.activation(logits_f_xs1), ys1)
            self.log_dict(scores_f, **self.log_params)

            scores_g = self.metric_dict_train_g_s(self.activation(logits_g_xs2), ys2)
            self.log_dict(scores_g, **self.log_params)

        return loss

    def loss_sup(
        self, logits_f_xs1: Tensor, logits_g_xs2: Tensor, ys1: Tensor, ys2: Tensor
    ) -> Tensor:
        log_probs_f_xs1 = self.log_activation(logits_f_xs1)
        log_probs_g_xs2 = self.log_activation(logits_g_xs2)
        return self.criterion_s(log_probs_f_xs1, ys1) + self.criterion_s(
            log_probs_g_xs2, ys2
        )

    def loss_cot(self, logits_f_xu: Tensor, logits_g_xu: Tensor) -> Tensor:
        probs_f_xu = self.activation(logits_f_xu)
        probs_g_xu = self.activation(logits_g_xu)

        mean_pred = 0.5 * (probs_f_xu + probs_g_xu)
        mean_pred = torch.clamp(mean_pred, min=1e-8)

        loss_mean_pred = mean_pred * mean_pred.log()
        loss_mean_pred = -loss_mean_pred.sum()

        loss_f = self.activation(logits_f_xu) * self.log_activation(logits_f_xu)
        loss_f = -loss_f.sum()

        loss_g = self.activation(logits_g_xu) * self.log_activation(logits_g_xu)
        loss_g = -loss_g.sum()

        bsize_u = logits_f_xu.shape[0]
        loss_cot = loss_mean_pred - 0.5 * (loss_f + loss_g) / bsize_u

        return loss_cot

    def loss_diff(
        self,
        logits_f_xs1: Tensor,
        logits_g_xs2: Tensor,
        logits_f_adv_g_xs2: Tensor,
        logits_g_adv_f_xs1: Tensor,
        logits_f_xu: Tensor,
        logits_g_xu: Tensor,
        logits_f_adv_g_xu: Tensor,
        logits_g_adv_f_xu: Tensor,
    ) -> Tensor:

        loss_g_xs2 = self.activation(logits_g_xs2) * self.log_activation(
            logits_f_adv_g_xs2
        )
        loss_g_xs2 = loss_g_xs2.sum()

        loss_f_xs1 = self.activation(logits_f_xs1) * self.log_activation(
            logits_g_adv_f_xs1
        )
        loss_f_xs1 = loss_f_xs1.sum()

        loss_g_xu = self.activation(logits_g_xu) * self.log_activation(
            logits_f_adv_g_xu
        )
        loss_g_xu = loss_g_xu.sum()

        loss_f_xu = self.activation(logits_f_xu) * self.log_activation(
            logits_g_adv_f_xu
        )
        loss_f_xu = loss_f_xu.sum()

        total_bsize = logits_f_xs1.shape[0] + logits_f_xu.shape[0]
        loss_diff = -(loss_g_xs2 + loss_f_xs1 + loss_g_xu + loss_f_xu) / total_bsize

        return loss_diff

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        xs, ys = batch
        probs_f_xs = self.activation(self.model_f(xs))
        probs_g_xs = self.activation(self.model_g(xs))

        self.log_dict(self.metric_dict_val_f(probs_f_xs, ys))
        self.log_dict(self.metric_dict_val_g(probs_g_xs, ys))

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        xs, ys = batch
        probs_f_xs = self.activation(self.model_f(xs))
        probs_g_xs = self.activation(self.model_g(xs))

        self.log_dict(self.metric_dict_test_f(probs_f_xs, ys))
        self.log_dict(self.metric_dict_test_g(probs_g_xs, ys))

    def forward(self, x: Tensor, model_used: str = "model_f") -> Tensor:
        """
        TODO: Default use model f, maybe use g ?
        """

        if model_used in ["f", "model_f"]:
            probs_x = self.activation(self.model_f(x))

        elif model_used in ["g", "model_g"]:
            probs_x = self.activation(self.model_g(x))

        elif model_used in ["mean"]:
            probs_f_x = self.activation(self.model_f(x))
            probs_g_x = self.activation(self.model_g(x))
            probs_x = (probs_f_x + probs_g_x) / 2.0

        elif model_used in ["most_confident"]:
            probs_f_x = self.activation(self.model_f(x))
            probs_g_x = self.activation(self.model_g(x))
            if probs_f_x.max() > probs_g_x.max():
                probs_x = probs_f_x
            else:
                probs_x = probs_g_x

        else:
            raise RuntimeError(
                f'Invalid model used "{model_used}". '
                f'Must be one of {("f", "model_f", "g", "model_g", "mean", "most_confident")}.'
            )

        return probs_x

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer

    def get_model_f(self) -> nn.Module:
        return self.model_f

    def get_model_g(self) -> nn.Module:
        return self.model_g
