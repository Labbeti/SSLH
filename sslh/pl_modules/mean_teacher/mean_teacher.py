#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple

import torch

from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Optimizer

from sslh.nn.ema import EMA
from sslh.nn.utils import ForwardDictAffix
from sslh.transforms.other.noise import GaussianNoise


class MeanTeacher(LightningModule):
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        optimizer: Optimizer,
        activation: nn.Module = nn.Softmax(dim=-1),
        criterion_s: nn.Module = nn.CrossEntropyLoss(),
        criterion_ccost: nn.Module = nn.MSELoss(),
        noise: nn.Module = GaussianNoise(),
        decay: float = 0.999,
        lambda_ccost: float = 1.0,
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
        for param in teacher.parameters():
            param.detach_()

        super().__init__()
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.activation = activation
        self.criterion_s = criterion_s
        self.criterion_ccost = criterion_ccost
        self.noise = noise
        self.decay = decay
        self.lambda_ccost = lambda_ccost

        self.metric_dict_train_s_stu = ForwardDictAffix(
            train_metrics, prefix="train/", suffix="_s_stu"
        )
        self.metric_dict_train_s_tea = ForwardDictAffix(
            train_metrics, prefix="train/", suffix="_s_tea"
        )
        self.metric_dict_train_u = ForwardDictAffix(
            train_metrics, prefix="train/", suffix="_pse_u"
        )
        self.metric_dict_val_stu = ForwardDictAffix(
            val_metrics, prefix="val/", suffix="_stu"
        )
        self.metric_dict_val_tea = ForwardDictAffix(
            val_metrics, prefix="val/", suffix="_tea"
        )
        self.metric_dict_test_stu = ForwardDictAffix(
            val_metrics, prefix="test/", suffix="_stu"
        )
        self.metric_dict_test_tea = ForwardDictAffix(
            val_metrics, prefix="test/", suffix="_tea"
        )

        self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)
        self.ema = EMA(teacher, decay, copy_model=False)

        self.save_hyperparameters(
            {
                "experiment": self.__class__.__name__,
                "model": student.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "activation": activation.__class__.__name__,
                "criterion_s": criterion_s.__class__.__name__,
                "criterion_ccost": criterion_ccost.__class__.__name__,
                "decay": decay,
                "lambda_ccost": lambda_ccost,
            }
        )
        self._cur_training_step = 0

    def training_step(
        self,
        batch: Tuple[Tuple[Tensor, Tensor], Tensor],
        batch_idx: int,
    ) -> Tensor:
        (xs, ys), xu = batch

        x = torch.cat((xs, xu))

        student_logits = self.student(x)
        bsize_s = xs.shape[0]
        student_logits_xs = student_logits[:bsize_s]
        student_probs = self.activation(student_logits)

        with torch.no_grad():
            noisy_x = self.noise(x)
            teacher_logits = self.teacher(noisy_x)
            teacher_probs = self.activation(teacher_logits)

        # Compute losses
        loss_s = self.criterion_s(student_logits_xs, ys.argmax(dim=-1))
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

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        xs, ys = batch
        student_probs_xs = self.activation(self.student(xs))
        teacher_probs_xs = self.activation(self.teacher(xs))

        self.log_dict(self.metric_dict_val_stu(student_probs_xs, ys))
        self.log_dict(self.metric_dict_val_tea(teacher_probs_xs, ys))

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        xs, ys = batch
        student_probs_xs = self.activation(self.student(xs))
        teacher_probs_xs = self.activation(self.teacher(xs))

        self.log_dict(self.metric_dict_test_stu(student_probs_xs, ys))
        self.log_dict(self.metric_dict_test_tea(teacher_probs_xs, ys))

    def forward(self, x: Tensor, probs_mode: str = "teacher") -> Tensor:
        """
        Note: Default use teacher
        """

        if probs_mode in ["tea", "teacher"]:
            probs_x = self.activation(self.teacher(x))

        elif probs_mode in ["stu", "student"]:
            probs_x = self.activation(self.student(x))

        elif probs_mode in ["mean"]:
            probs_f_x = self.activation(self.teacher(x))
            probs_g_x = self.activation(self.student(x))
            probs_x = (probs_f_x + probs_g_x) / 2.0

        elif probs_mode in ["most_confident"]:
            probs_f_x = self.activation(self.teacher(x))
            probs_g_x = self.activation(self.student(x))
            if probs_f_x.max() > probs_g_x.max():
                probs_x = probs_f_x
            else:
                probs_x = probs_g_x

        else:
            PRED_MODES = ("tea", "teacher", "stu", "student", "mean", "most_confident")
            raise RuntimeError(
                f"Invalid argument {probs_mode=}. Must be one of {PRED_MODES}."
            )

        return probs_x

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer
