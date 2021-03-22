
from torch.nn import Module, BCELoss, LogSigmoid, LogSoftmax, Sigmoid, Softmax, Sequential
from torch.optim import Optimizer, Adam, SGD
from typing import Callable, Optional

from mlu.nn import CrossEntropyWithVectors, BCELossBatchMean, Clamp, JSDivLoss, KLDivLossWithProbabilities, Identity
from sslh.callbacks.schedulers import CosineScheduler, SoftCosineScheduler, LRSchedulerCallback


def get_criterion_from_name(name: str, reduction: str = "mean") -> Module:
	name = name.lower()
	reduction = reduction.lower()

	if name in ["CrossEntropyWithVectors".lower(), "CrossEntropy".lower(), "ce".lower()]:
		loss = CrossEntropyWithVectors(reduction=reduction)
	elif name in ["BCELoss".lower(), "bce".lower()]:
		if reduction == "none":
			loss = BCELossBatchMean()
		else:
			loss = BCELoss(reduction=reduction)
	elif name in ["JSDivLoss".lower(), "js".lower()]:
		loss = JSDivLoss(reduction=reduction)
	elif name in ["KLDivLoss".lower(), "kl".lower()]:
		loss = KLDivLossWithProbabilities(reduction=reduction)
	else:
		raise NotImplementedError(
			f"Unknown criterion name '{name}'. Must be one of "
			f"{('CrossEntropyWithVectors', 'CrossEntropy', 'ce', 'BCELoss', 'bce', 'JSDivLoss', 'js', 'KLDivLoss', 'kl')}."
		)

	return loss


def get_optimizer_from_name(name: str, model: Module, **kwargs) -> Optimizer:
	"""
		Instantiate optimizer from args and torch module.
		Available optimizers : Adam, SGD.

		:param name: TODO
		:param model: The torch module to update with the optimizer.
		:returns: The optimizer to use.
	"""
	name = name.lower()

	if name == "Adam".lower():
		optimizer = Adam(model.parameters(), **kwargs)
	elif name == "SGD".lower():
		optimizer = SGD(model.parameters(), **kwargs)
	else:
		raise RuntimeError(f"Unknown optimizer '{name}'. Must be one of {('Adam', 'SGD')}")

	return optimizer


def get_scheduler_from_name(name: str, optimizer: Optimizer, **kwargs) -> Optional[LRSchedulerCallback]:
	"""
		Instantiate scheduler from args and optimizer.
		Available schedulers : CosineLRScheduler, MultiStepLR and SoftCosineLRScheduler.

		:param name: TODO
		:param optimizer: The torch Optimizer to update with the scheduler.
		:returns: The scheduler to use. If "args.scheduler" is not an available scheduler then returns None.
	"""
	name = str(name).lower()

	if name in ["CosineScheduler".lower(), "Cosine".lower()]:
		scheduler = CosineScheduler(optimizer, **kwargs)
	elif name in ["SoftCosineScheduler".lower(), "SoftCosine".lower()]:
		scheduler = SoftCosineScheduler(optimizer, **kwargs)
	else:
		raise RuntimeError(
			f"Unknown scheduler name '{name}'. "
			f"Must be one of {('CosineScheduler', 'Cosine', 'SoftCosineScheduler', 'SoftCosine')}."
		)

	return scheduler


def get_activation_from_name(name: str, dim: int = -1, clamp: bool = True, clamp_min: float = 2e-30) -> Callable:
	"""
		Build an activation function.

		:param name: The name of the activation function. Can be "softmax" or "sigmoid".
		:param dim: The dimension to apply the activation function. (default: -1)
		:param clamp: If True, add a clamp operation after the activation. (default: True)
		:param clamp_min: The minimal value of the clamp. Unused if clamp == False. (default: 2e-30)
		:return: The activation function.
	"""
	name = str(name).lower()

	if name == "softmax":
		activation = Softmax(dim=dim)
	elif name == "sigmoid":
		activation = Sigmoid()
	elif name == "log_softmax":
		activation = LogSoftmax(dim=dim)
	elif name == "log_sigmoid":
		activation = LogSigmoid()
	elif name in ["none", "identity"]:
		activation = Identity()
	else:
		raise RuntimeError(
			f"Unknown activation function '{name}'. "
			f"Must be one of {('softmax', 'sigmoid', 'log_softmax', 'log_sigmoid', 'none', 'identity')}."
		)

	if clamp:
		activation = Sequential(activation, Clamp(min_=clamp_min))

	return activation
