
import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, MSELoss, Softmax
from torch.optim import Optimizer
from typing import Optional, Tuple

from mlu.metrics import MetricDict
from mlu.nn import CrossEntropyWithVectors
from mlu.nn.ema import EMA


class MeanTeacher(LightningModule):
	def __init__(
		self,
		student: Module,
		teacher: Module,
		optimizer: Optimizer,
		activation: Module = Softmax(dim=-1),
		criterion_s: Module = CrossEntropyWithVectors(),
		criterion_ccost: Module = MSELoss(),
		decay: float = 0.999,
		lambda_ccost: float = 1.0,
		metric_dict_train_s_stu: Optional[MetricDict] = None,
		metric_dict_train_s_tea: Optional[MetricDict] = None,
		metric_dict_train_u: Optional[MetricDict] = None,
		metric_dict_val_stu: Optional[MetricDict] = None,
		metric_dict_val_tea: Optional[MetricDict] = None,
		metric_dict_test_stu: Optional[MetricDict] = None,
		metric_dict_test_tea: Optional[MetricDict] = None,
		log_on_epoch: bool = True,
	):
		if metric_dict_train_s_stu is None:
			metric_dict_train_s_stu = MetricDict()

		if metric_dict_train_s_tea is None:
			metric_dict_train_s_tea = MetricDict()

		if metric_dict_train_u is None:
			metric_dict_train_u = MetricDict()

		if metric_dict_val_stu is None:
			metric_dict_val_stu = MetricDict()

		if metric_dict_val_tea is None:
			metric_dict_val_tea = MetricDict()

		if metric_dict_test_stu is None:
			metric_dict_test_stu = MetricDict()

		if metric_dict_test_tea is None:
			metric_dict_test_tea = MetricDict()

		super().__init__()
		self.student = student
		self.teacher = teacher
		self.optimizer = optimizer
		self.activation = activation
		self.criterion_s = criterion_s
		self.criterion_ccost = criterion_ccost
		self.decay = decay
		self.lambda_ccost = lambda_ccost
		self.metric_dict_train_s_stu = metric_dict_train_s_stu
		self.metric_dict_train_s_tea = metric_dict_train_s_tea
		self.metric_dict_train_u = metric_dict_train_u
		self.metric_dict_val_stu = metric_dict_val_stu
		self.metric_dict_val_tea = metric_dict_val_tea
		self.metric_dict_test_stu = metric_dict_test_stu
		self.metric_dict_test_tea = metric_dict_test_tea
		self.log_on_epoch = log_on_epoch

		self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)

		for param in teacher.parameters():
			param.detach_()
		teacher.eval()

		self.ema = EMA(teacher, decay, False)

		self.save_hyperparameters({
			"experiment": self.__class__.__name__,
			"model": student.__class__.__name__,
			"optimizer": optimizer.__class__.__name__,
			"activation": activation.__class__.__name__,
			"criterion_s": criterion_s.__class__.__name__,
			"criterion_ccost": criterion_ccost.__class__.__name__,
			"decay": decay,
			"lambda_ccost": lambda_ccost,
		})

	def training_step(
		self,
		batch: Tuple[Tuple[Tensor, Tensor], Tensor],
		batch_idx: int,
	):
		(xs, ys), xu = batch
		bsize_s = xs.shape[0]

		pred_student_xs = self.activation(self.student(xs))

		x = torch.cat((xs, xu))
		pred_student = self.activation(self.student(x))
		with torch.no_grad():
			pred_teacher = self.activation(self.teacher(x))

		# Compute losses
		loss_s = self.criterion_s(pred_student_xs, ys)
		loss_ccost = self.criterion_ccost(pred_student, pred_teacher)
		loss = loss_s + self.lambda_ccost * loss_ccost

		with torch.no_grad():
			# Compute metrics
			self.log_dict({"loss": loss, "loss_s": loss_s, "loss_ccost": loss_ccost})

			pred_teacher_xs, pred_teacher_xu = pred_teacher[:bsize_s], pred_teacher[bsize_s:]
			pred_student_xu = pred_student[bsize_s:]

			self.log_dict(self.metric_dict_train_s_stu(pred_student_xs, ys), **self.log_params)
			self.log_dict(self.metric_dict_train_s_tea(pred_teacher_xs, ys), **self.log_params)
			self.log_dict(self.metric_dict_train_u(pred_student_xu, pred_teacher_xu), **self.log_params)

			# Update teacher with Exponential Moving Average
			self.ema.decay = min(1.0 - 1.0 / (self.current_epoch + 1), self.ema.decay)
			self.ema.update(self.student)

		return loss

	def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_student_xs = self.activation(self.student(xs))
		pred_teacher_xs = self.activation(self.teacher(xs))

		self.log_dict(self.metric_dict_val_stu(pred_student_xs, ys))
		self.log_dict(self.metric_dict_val_tea(pred_teacher_xs, ys))

	def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
		xs, ys = batch
		pred_student_xs = self.activation(self.student(xs))
		pred_teacher_xs = self.activation(self.teacher(xs))

		self.log_dict(self.metric_dict_test_stu(pred_student_xs, ys))
		self.log_dict(self.metric_dict_test_tea(pred_teacher_xs, ys))

	def forward(self, x: Tensor, model_used: str = "teacher") -> Tensor:
		"""
			TODO: Default use teacher, maybe use student ?
		"""

		if model_used in ["tea", "teacher"]:
			pred_x = self.activation(self.teacher(x))

		elif model_used in ["stu", "student"]:
			pred_x = self.activation(self.student(x))

		elif model_used in ["mean"]:
			pred_f_x = self.activation(self.teacher(x))
			pred_g_x = self.activation(self.student(x))
			pred_x = (pred_f_x + pred_g_x) / 2.0

		elif model_used in ["most_confident"]:
			pred_f_x = self.activation(self.teacher(x))
			pred_g_x = self.activation(self.student(x))
			if pred_f_x.max() > pred_g_x.max():
				pred_x = pred_f_x
			else:
				pred_x = pred_g_x

		else:
			raise RuntimeError(
				f"Invalid model used '{model_used}'. "
				f"Must be one of {('tea', 'teacher', 'stu', 'student', 'mean', 'most_confident')}."
			)

		return pred_x

	def configure_optimizers(self) -> Optimizer:
		return self.optimizer

	def load_state_dict(self, *args, **kwargs):
		self.ema = None
		super().load_state_dict(*args, **kwargs)
		self.ema = EMA(self.teacher, self.decay, False)
