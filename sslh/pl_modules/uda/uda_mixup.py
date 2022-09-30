#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple

import torch

from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim.optimizer import Optimizer

from sslh.pl_modules.uda.uda import UDA
from sslh.nn.loss import NLLLossVecTargets
from sslh.transforms.other.mixup import MixUpModule


class UDAMixUp(UDA):
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
        alpha: float = 0.75,
        train_metrics: Optional[Dict[str, Module]] = None,
        val_metrics: Optional[Dict[str, Module]] = None,
        log_on_epoch: bool = True,
    ):
        """
        Unsupervised Data Augmentation with MixUp (UDAM) LightningModule.

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
            criterion_s=criterion_s,
            criterion_u=criterion_u,
            lambda_u=lambda_u,
            threshold=threshold,
            temperature=temperature,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            log_on_epoch=log_on_epoch,
        )
        self.alpha = alpha
        self.mixup = MixUpModule(alpha=alpha, apply_max=True)

        self.save_hyperparameters({"alpha": alpha})

    def training_step(
        self,
        batch: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]],
        batch_idx: int,
    ):
        (xs, ys), (xu, xu_strong) = batch

        with torch.no_grad():
            # Compute pseudo-labels 'yu' and mask
            yu, mask = self.guess_label_and_mask(xu)

            xs_mix, ys_mix = self.mixup(xs, xu, ys, yu)
            xu_mix, yu_mix = self.mixup(xu_strong, xs, yu, ys)

        # Compute predictions on xs and xu
        probs_xs_mix = self.activation(self.model(xs_mix))
        probs_xu_mix = self.activation(self.model(xu_mix))

        # Criterion (loss_s of shape bsize_s, loss_u of shape bsize_u)
        loss_s = self.criterion_s(probs_xs_mix, ys_mix)
        loss_u = self.criterion_u(probs_xu_mix, yu_mix)

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

            probs_xs = self.activation(self.model(xs))
            scores_s = self.metric_dict_train_s(probs_xs, ys)
            self.log_dict(scores_s, **self.log_params)

            probs_xu = self.activation(self.model(xu))
            scores_u = self.metric_dict_train_u_pseudo(probs_xu, yu)
            self.log_dict(scores_u, **self.log_params)

        return loss
