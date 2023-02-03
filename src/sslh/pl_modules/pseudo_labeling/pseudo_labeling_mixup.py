#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from typing import Dict, Optional, Tuple

from sslh.nn.labels import OneHot
from sslh.nn.loss import CrossEntropyLossVecTargets
from sslh.pl_modules.pseudo_labeling.pseudo_labeling import PseudoLabeling
from sslh.transforms.other.mixup import MixupModule


class PseudoLabelingMixup(PseudoLabeling):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        activation: nn.Module = nn.Softmax(dim=-1),
        criterion_s: nn.Module = CrossEntropyLossVecTargets(reduction="none"),
        criterion_u: nn.Module = CrossEntropyLossVecTargets(reduction="none"),
        target_transform: nn.Module = OneHot(n_classes=10),
        lambda_u: float = 1.0,
        threshold: float = 0.0,
        alpha: float = 0.75,
        train_metrics: Optional[Dict[str, nn.Module]] = None,
        val_metrics: Optional[Dict[str, nn.Module]] = None,
        log_on_epoch: bool = True,
    ) -> None:
        """
        Pseudo Labeling with Mixup (PLM) LightningModule.

        :param model: The PyTorch Module to train.
                The forward() must return logits to classify the data.
        :param optimizer: The PyTorch optimizer to use.
        :param activation: The activation function of the model.
                (default: Softmax(dim=-1))
        :param criterion_s: The loss component 'L_s' of PL.
                (default: CrossEntropyWithVectors())
        :param criterion_u: The loss component 'L_u' of PL.
                (default: CrossEntropyWithVectors())
        :param target_transform: The target transform for convert binarized labels to vector probabilities.
                (default: OneHot(n_classes=10))
        :param lambda_u: The coefficient of the 'L_u' component.
                (default: 1.0)
        :param threshold: The confidence threshold 'tau' used for the mask of the 'L_u' component.
                (default: 0.95)
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
            target_transform=target_transform,
            lambda_u=lambda_u,
            threshold=threshold,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            log_on_epoch=log_on_epoch,
        )
        self.alpha = alpha
        self.mixup = MixupModule(alpha=alpha, apply_max=True)

        self.save_hyperparameters({"alpha": alpha})

    def training_step(
        self,
        batch: Tuple[Tuple[Tensor, Tensor], Tensor],
        batch_idx: int,
    ) -> Tensor:
        (xs, ys), xu = batch

        with torch.no_grad():
            # Compute pseudo-labels 'yu' and mask
            yu, mask = self.guess_label_and_mask(xu)

            xs_mix, ys_mix = self.mixup(xs, xu, ys, yu)
            xu_mix, yu_mix = self.mixup(xu, xs, yu, ys)

        # Compute predictions on xs and xu
        logits_xs_mix = self.model(xs_mix)
        logits_xu_mix = self.model(xu_mix)

        # Criterion (loss_s of shape bsize_s, loss_u of shape bsize_u)
        loss_s = self.criterion_s(logits_xs_mix, ys_mix)
        loss_u = self.criterion_u(logits_xu_mix, yu_mix)

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
