#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple

import torch

from torch import nn, Tensor
from torch.optim.optimizer import Optimizer

from sslh.nn.loss import CrossEntropyLossVecTargets
from sslh.pl_modules.fixmatch.fixmatch import FixMatch


class FixMatchThresholdGuess(FixMatch):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        activation: nn.Module = nn.Softmax(dim=-1),
        criterion_s: nn.Module = CrossEntropyLossVecTargets(reduction="none"),
        criterion_u: nn.Module = CrossEntropyLossVecTargets(reduction="none"),
        lambda_u: float = 1.0,
        threshold: float = 0.0,
        threshold_guess: float = 0.75,
        train_metrics: Optional[Dict[str, nn.Module]] = None,
        val_metrics: Optional[Dict[str, nn.Module]] = None,
        log_on_epoch: bool = True,
    ) -> None:
        """
        FixMatch with Threshold Guess pseudo label (FMTG) LightningModule.

        :param model: The PyTorch nn.Module to train.
                The forward() must return logits to classify the data.
        :param optimizer: The PyTorch optimizer to use.
        :param activation: The activation function of the model.
                (default: Softmax(dim=-1))
        :param criterion_s: The loss component 'L_s' of FM.
                (default: CrossEntropyWithVectors())
        :param criterion_u: The loss component 'L_u' of FM.
                (default: CrossEntropyWithVectors())
        :param lambda_u: The coefficient of the 'L_u' component.
                (default: 1.0)
        :param threshold: The confidence threshold 'tau' used for the mask of the 'L_u' component.
                (default: 0.0)
        :param threshold_guess: The threshold used for binarize to multihot labels.
                (default: 0.75)
        :param train_metrics: An optional dictionary of metrics nn.Modules for training.
                (default: None)
        :param val_metrics: An optional dictionary of metrics nn.Modules for validation.
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
            target_transform=nn.Identity(),
            lambda_u=lambda_u,
            threshold=threshold,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            log_on_epoch=log_on_epoch,
        )
        self.threshold_guess = threshold_guess

        self.save_hyperparameters({"threshold_guess": threshold_guess})

    @torch.no_grad()
    def guess_label_and_mask(self, xu_weak: Tensor) -> Tuple[Tensor, Tensor]:
        probs_xu_weak = self.activation(self.model(xu_weak))
        yu = probs_xu_weak.ge(self.threshold_guess).to(probs_xu_weak.dtype)
        probabilities_max, _ = probs_xu_weak.max(dim=-1)
        mask = probabilities_max.ge(self.threshold).to(probs_xu_weak.dtype)
        return yu, mask
