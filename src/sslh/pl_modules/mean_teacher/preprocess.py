#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Tuple

from torch import nn


class MeanTeacherAugPreProcess(nn.Module):
    def __init__(
        self, student_transform: Callable, teacher_transform: Callable
    ) -> None:
        super().__init__()
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform

    def forward(self, data: Any) -> Tuple[Any, Any]:
        return self.student_transform(data), self.teacher_transform(data)
