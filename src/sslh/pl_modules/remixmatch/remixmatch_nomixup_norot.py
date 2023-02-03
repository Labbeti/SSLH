#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Tuple

import torch

from torch import nn, Tensor
from torch.optim import Optimizer

from sslh.nn.loss import CrossEntropyLossVecTargets
from sslh.pl_modules.remixmatch.remixmatch import ReMixMatch


class ReMixMatchNoMixupNoRot(ReMixMatch):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        activation: nn.Module = nn.Softmax(dim=-1),
        criterion_s: nn.Module = CrossEntropyLossVecTargets(),
        criterion_u: nn.Module = CrossEntropyLossVecTargets(),
        criterion_u1: nn.Module = CrossEntropyLossVecTargets(),
        lambda_u: float = 1.5,
        lambda_u1: float = 0.5,
        n_augms: int = 2,
        temperature: float = 0.5,
        history: int = 128,
        train_metrics: Optional[Dict[str, nn.Module]] = None,
        val_metrics: Optional[Dict[str, nn.Module]] = None,
        log_on_epoch: bool = True,
    ) -> None:
        """
        ReMixMatch without Mixup and without self-supervised loss component LightningModule.

        :param model: The PyTorch nn.Module to train.
                The forward() must return logits to classify the data.
                The forward_rot() method must return logits for classify the self-transform applied.
        :param optimizer: The PyTorch optimizer to use.
        :param activation: The activation function of the model.
                (default: nn.Softmax(dim=-1))
        :param criterion_s: The loss component 'L_s' of RMM.
                (default: CrossEntropyWithVectors())
        :param criterion_u: The loss component 'L_u' of RMM.
                (default: CrossEntropyWithVectors())
        :param criterion_u1: The loss component 'L_u1' of RMM.
                (default: CrossEntropyWithVectors())
        :param lambda_u: The coefficient of the 'L_u' component. (default: 1.5)
        :param lambda_u1: The coefficient of the 'L_u1' component. (default: 0.5)
        :param n_augms: The number of strong augmentations applied. (default: 2)
        :param temperature: The temperature applied by the sharpen function.
                A lower temperature make the pseudo-label produced more 'one-hot'.
                (default: 0.5)
        :param history: The number of batches labels to keep for compute the labeled and unlabeled classes distributions.
                (default: 128)
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
            activation_r=nn.Identity(),
            criterion_s=criterion_s,
            criterion_u=criterion_u,
            criterion_u1=criterion_u1,
            criterion_r=nn.Identity(),
            self_transform=nn.Identity(),
            lambda_u=lambda_u,
            lambda_u1=lambda_u1,
            lambda_r=0.0,
            n_augms=n_augms,
            temperature=temperature,
            alpha=0.0,
            history=history,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            train_metrics_r=None,
            log_on_epoch=log_on_epoch,
            check_model=False,
        )

    def training_step(
        self,
        batch: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]],
        batch_idx: int,
    ) -> Tensor:
        (xs_strong, ys), (xu_weak, xu_strong_lst) = batch

        with torch.no_grad():
            probs_xu_weak = self.model(xu_weak)

            # Update labeled and unlabeled classes distributions
            self.average_probs_s.add_pred(ys)
            self.average_probs_u.add_pred(probs_xu_weak)

            # Apply distribution alignment
            yu = (
                probs_xu_weak
                * self.average_probs_s.get_mean()
                / self.average_probs_u.get_mean()
            )
            yu = yu / yu.norm(p=1, dim=-1, keepdim=True)

            # Sharpen & duplicate pseudo-label for each augmented variants of xu
            yu = self.sharpen(yu)
            yu_lst = yu.repeat([self.n_augms + 1] + [1] * (len(yu.shape) - 1))

            xu_lst = [xu_weak] + xu_strong_lst
            xu_lst = torch.vstack(xu_lst)

            # Get the batch xu1 for 'L_u1' component and self-supervised transform
            xu1_strong = xu_strong_lst[0].clone()
            yu1 = yu

        logits_xs_strong = self.model(xs_strong)
        logits_xu_weak_and_strong = self.model(xu_lst)
        logits_xu1 = self.model(xu1_strong)

        loss_s = self.criterion_s(logits_xs_strong, ys)
        loss_u = self.criterion_u(logits_xu_weak_and_strong, yu_lst)
        loss_u1 = self.criterion_u1(logits_xu1, yu1)

        loss = loss_s + self.lambda_u * loss_u + self.lambda_u1 * loss_u1

        with torch.no_grad():
            # Compute metrics
            scores = {
                "loss": loss,
                "loss_s": loss_s,
                "loss_u": loss_u,
                "loss_u1": loss_u1,
            }
            scores = {f"train/{k}": v.cpu() for k, v in scores.items()}
            self.log_dict(scores, **self.log_params)

            probs_xs_strong = self.activation(logits_xs_strong)
            scores_s = self.metric_dict_train_s(probs_xs_strong, ys)
            self.log_dict(scores_s, **self.log_params)

            probs_xu_weak_and_strong = self.activation(logits_xu_weak_and_strong)
            scores_u = self.metric_dict_train_u_pseudo(probs_xu_weak_and_strong, yu_lst)
            self.log_dict(scores_u, **self.log_params)

            probs_xu1 = self.activation(logits_xu1)
            scores_u1 = self.metric_dict_train_u_pseudo(probs_xu1, yu1)
            self.log_dict(scores_u1, **self.log_params)

        return loss
