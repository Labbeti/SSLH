
import math

from torch.nn import Module, BCELoss, LogSigmoid, LogSoftmax, Sigmoid, Softmax, Sequential, Parameter, MSELoss, BCEWithLogitsLoss
from torch.optim import Optimizer, Adam, SGD
from typing import Iterator, Union

from mlu.nn import CrossEntropyWithVectors, BCELossBatchMean, Clamp, JSDivLoss, KLDivLossWithProbabilities, Identity
from sslh.callbacks.schedulers import CosineScheduler, SoftCosineScheduler, LRSchedulerCallback


def get_criterion_from_name(name: str, reduction: str = 'mean', **kwargs) -> Module:
	"""
		Return a criterion Module with a specific name.

		:param name: The name of the criterion.
		:param reduction: The reduction function to apply to losses outputs. (default: 'mean')
		:return: The criterion as torch Module.
	"""
	name = name.lower()
	reduction = reduction.lower()

	if name in ['CrossEntropyWithVectors'.lower(), 'CrossEntropy'.lower(), 'CE'.lower()]:
		loss = CrossEntropyWithVectors(reduction=reduction, **kwargs)

	elif name in ['MSELoss'.lower(), 'MSE'.lower()]:
		loss = MSELoss(reduction=reduction, **kwargs)

	elif name in ['BCELoss'.lower(), 'BCE'.lower()]:
		if reduction == 'none':
			loss = BCELossBatchMean(**kwargs)
		else:
			loss = BCELoss(reduction=reduction, **kwargs)

	elif name in ['BCEWithLogitsLoss'.lower(), 'BCEWithLogits'.lower(), 'BCELogits'.lower()]:
		loss = BCEWithLogitsLoss(reduction=reduction, **kwargs)

	elif name in ['JSDivLoss'.lower(), 'JS'.lower()]:
		loss = JSDivLoss(reduction=reduction, **kwargs)

	elif name in ['KLDivLoss'.lower(), 'KL'.lower()]:
		loss = KLDivLossWithProbabilities(reduction=reduction, **kwargs)

	else:
		raise NotImplementedError(
			f'Unknown criterion name "{name}". Must be one of '
			f'("CrossEntropyWithVectors", "CrossEntropy", "ce", "BCELoss", "bce", "BCEWithLogitsLoss", "BCEWithLogits", '
			f'"BCELogits", "JSDivLoss", "js", "KLDivLoss", "kl").'
		)

	return loss


def get_optimizer_from_name(name: str, parameters: Union[Module, Iterator[Parameter]], **kwargs) -> Optimizer:
	"""
		Instantiate optimizer from args and torch module.
		Available optimizers : Adam, SGD.

		:param name: The name of the optimizer.
		:param parameters: The torch parameters or module to update with the optimizer.
		:param kwargs: The keyword arguments used for build the optimizer.
		:returns: The optimizer to use as torch Optimizer.
	"""
	name = name.lower()

	if isinstance(parameters, Module):
		parameters = parameters.parameters()

	if name == 'Adam'.lower():
		optimizer = Adam(parameters, **kwargs)
	elif name == 'SGD'.lower():
		optimizer = SGD(parameters, **kwargs)
	else:
		raise RuntimeError(f'Unknown optimizer "{name}". Must be one of {("Adam", "SGD")}.')

	return optimizer


def get_scheduler_from_name(name: str, optimizer: Optimizer, **kwargs) -> LRSchedulerCallback:
	"""
		Instantiate scheduler from args and optimizer.

		Available schedulers are :
			- CosineScheduler
			- SoftCosineScheduler

		:param name: The name of the scheduler.
		:param optimizer: The torch Optimizer to update with the scheduler.
		:param kwargs: The keyword arguments for build the scheduler.
		:returns: The scheduler callback built.
	"""
	name = str(name).lower()

	if name in ['CosineScheduler'.lower(), 'Cosine'.lower()]:
		scheduler = CosineScheduler(optimizer, **kwargs)
	elif name in ['SoftCosineScheduler'.lower(), 'SoftCosine'.lower()]:
		scheduler = SoftCosineScheduler(optimizer, **kwargs)
	else:
		raise RuntimeError(
			f'Unknown scheduler name "{name}". '
			f'Must be one of {("CosineScheduler", "Cosine", "SoftCosineScheduler", "SoftCosine")}.'
		)

	return scheduler


def get_activation_from_name(
	name: str,
	dim: int = -1,
	clamp_min: float = 2e-30,
	clamp_max: float = 1.0 - 2e-7,
) -> Module:
	"""
		Build an activation function.

		:param name: The name of the activation function.
			Can be 'softmax', 'sigmoid', 'log_softmax', 'log_sigmoid' or 'identity'.
		:param dim: The dimension to apply the activation function.
			(default: -1)
		:param clamp_min: The minimal value of the clamp.
			(default: 2e-30)
		:param clamp_max: The maximal value of the clamp.
			(default: (1.0 - 2e-7))
		:return: The activation function as a torch Module.
	"""
	name = str(name).lower()

	if name == 'softmax':
		activation = Softmax(dim=dim)
	elif name == 'sigmoid':
		activation = Sigmoid()
	elif name == 'log_softmax':
		activation = LogSoftmax(dim=dim)
	elif name == 'log_sigmoid':
		activation = LogSigmoid()
	elif name == 'identity':
		activation = Identity()
	else:
		raise RuntimeError(
			f'Unknown activation function "{name}". '
			f'Must be one of {("softmax", "sigmoid", "log_softmax", "log_sigmoid", "identity")}.'
		)

	if not math.isinf(clamp_min) or not math.isinf(clamp_max):
		activation = Sequential(activation, Clamp(clamp_min, clamp_max))

	return activation
