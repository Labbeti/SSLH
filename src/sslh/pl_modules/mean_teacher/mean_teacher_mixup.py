#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Dict, Optional, Tuple

import torch

from torch import nn, Tensor
from torch.optim import Optimizer

from sslh.nn.loss import CrossEntropyLossVecTargets
from sslh.pl_modules.mean_teacher.mean_teacher import MeanTeacher
from sslh.transforms.other.mixup import MixupModule


class MeanTeacherMixup(MeanTeacher):
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        optimizer: Optimizer,
        activation: nn.Module = nn.Softmax(dim=-1),
        criterion_s: nn.Module = CrossEntropyLossVecTargets(),
        criterion_ccost: nn.Module = nn.MSELoss(),
        noise: Callable = nn.Identity(),
        decay: float = 0.999,
        lambda_ccost: float = 1.0,
        buffers_mode: str = "none",
        alpha: float = 0.75,
        train_metrics: Optional[Dict[str, nn.Module]] = None,
        val_metrics: Optional[Dict[str, nn.Module]] = None,
        log_on_epoch: bool = True,
    ) -> None:
        """
        Mean Teacher (MT) LightningModule.

        :param student: The student model optimized.
        :param teacher: The teacher model updated by EMA with student.
        :param optimizer: The pytorch optimizer for the student.
        :param activation: The activation function of the model.
                (default: Softmax(dim=-1))
        :param criterion_s: The supervised criterion used for compute 'L_s'.
                (default: CrossEntropyWithVectors(log_input=True))
        :param criterion_ccost: The consistency cost criterion for compute 'L_cot'.
                (default: MSELoss())
        :param noise: TODO
        :param decay: The decay hyperparameter used for update the teacher model.
                (default: 0.999)
        :param lambda_ccost: The coefficient for consistency cost loss component.
                (default: 1.0)
        :param train_metrics: An optional dictionary of metrics modules for training.
                (default: None)
        :param val_metrics: An optional dictionary of metrics modules for validation.
                (default: None)
        :param log_on_epoch: If True, log only the epoch means of each train metric score.
                (default: True)
        """
        super().__init__(
            student,
            teacher,
            optimizer,
            activation,
            criterion_s,
            criterion_ccost,
            noise,
            decay,
            lambda_ccost,
            buffers_mode,
            train_metrics,
            val_metrics,
            log_on_epoch,
        )
        self.save_hyperparameters(
            {
                "alpha": alpha,
            }
        )
        self.mixup = MixupModule(alpha, apply_max=True)

    def training_step(
        self,
        batch: Tuple[Tuple[Tuple[Tensor, Tensor], Tensor], Tuple[Tensor, Tensor]],
        batch_idx: int,
    ) -> Tensor:
        ((student_xs, teacher_xs), ys), (student_xu, teacher_xu) = batch
        bsize_s = student_xs.shape[0]

        with torch.no_grad():
            if self.ema.buffers_mode == "ema":
                self.teacher.eval()
            teacher_x = torch.cat((teacher_xs, teacher_xu))
            teacher_noisy_x = self.noise(teacher_x)
            teacher_logits = self.teacher(teacher_noisy_x)
            teacher_probs = self.activation(teacher_logits)
            teacher_probs_u = teacher_probs[bsize_s:]

            student_xs, ys = self.mixup(student_xs, student_xu, ys, teacher_probs_u)
            student_xu, teacher_probs_u = self.mixup(
                student_xu, student_xs, teacher_probs_u, ys
            )
            teacher_probs[bsize_s:] = teacher_probs_u

        student_x = torch.cat((student_xs, student_xu))
        student_logits = self.student(student_x)
        student_probs = self.activation(student_logits)
        student_logits_xs = student_logits[:bsize_s]

        # Compute losses
        loss_s = self.criterion_s(student_logits_xs, ys)
        loss_ccost = self.criterion_ccost(student_probs, teacher_probs)
        loss = loss_s + self.lambda_ccost * loss_ccost

        with torch.no_grad():
            # Compute metrics
            scores = {
                "loss": loss,
                "loss_s": loss_s,
                "loss_ccost": loss_ccost,
                "ema_alpha": torch.as_tensor(
                    self.ema.get_cur_alpha(self._cur_training_step)
                ),
            }
            scores = {f"train/{k}": v.cpu() for k, v in scores.items()}
            self.log_dict(scores, **self.log_params)

            student_probs_xs = student_probs[:bsize_s]
            student_probs_xu = student_probs[bsize_s:]
            teacher_probs_xs = teacher_probs[:bsize_s]
            teacher_probs_xu = teacher_probs[bsize_s:]

            self.log_dict(
                self.metric_dict_train_s_stu(student_probs_xs, ys), **self.log_params
            )
            self.log_dict(
                self.metric_dict_train_s_tea(teacher_probs_xs, ys), **self.log_params
            )
            self.log_dict(
                self.metric_dict_train_u(student_probs_xu, teacher_probs_xu),
                **self.log_params,
            )

            # Update teacher with Exponential Moving Average
            self.ema.ema_update(self.student, self._cur_training_step)
            self._cur_training_step += 1

        return loss
