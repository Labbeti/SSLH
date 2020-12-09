import math

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


Scheduler_t = "Scheduler"


class CosineLRScheduler(LambdaLR):
	"""
		Scheduler that decreases the learning rate from lr0 to almost 0 by using the following rule :
		lr = lr0 * cos(7 * pi * epoch / (16 * nb_epochs))
	"""
	def __init__(self, optim: Optimizer, nb_epochs: int):
		self.nb_epochs = nb_epochs
		super().__init__(optim, self.lr_lambda)

	def lr_lambda(self, epoch: int) -> float:
		return math.cos(7.0 * math.pi * epoch / (16.0 * self.nb_epochs))


class SoftCosineLRScheduler(LambdaLR):
	"""
		Scheduler that decreases the learning rate from lr0 to almost 0 by using the following rule :
		lr = lr0 * (1 + np.cos((epoch - 1) * pi / nb_epochs)) * 0.5
	"""
	def __init__(self, optim: Optimizer, nb_epochs: int):
		self.nb_epochs = nb_epochs
		super().__init__(optim, self.lr_lambda)

	def lr_lambda(self, epoch: int) -> float:
		return (1.0 + math.cos((epoch - 1) * math.pi / self.nb_epochs)) * 0.5
