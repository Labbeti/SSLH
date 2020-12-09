
import torch

from torch import Tensor
from torch.nn import Module, KLDivLoss, LogSoftmax, CrossEntropyLoss
from torch.nn.functional import softplus
from torch.optim.optimizer import Optimizer
from typing import Callable, List, Optional


DEFAULT_EPSILON = 2e-20


def normalized(batch: Tensor, dim: int) -> Tensor:
	"""
		Return the vector normalized.
	"""
	return batch / batch.norm(p=1, dim=dim, keepdim=True)


def same_shuffle(values: List[Tensor]) -> List[Tensor]:
	"""
		Shuffle each value of values with the same indexes.
	"""
	indices = torch.randperm(len(values[0]))
	for i in range(len(values)):
		values[i] = values[i][indices]
	return values


def merge_first_dimension(t: Tensor) -> Tensor:
	"""
		Reshape tensor of size (M, N, ...) to (M*N, ...).
	"""
	shape = list(t.shape)
	if len(shape) < 2:
		raise RuntimeError("Invalid nb of dimension ({:d}) for merge_first_dimension. Should have at least 2 dimensions.".format(len(shape)))
	return t.reshape(shape[0] * shape[1], *shape[2:])


def get_lrs(optim: Optimizer) -> List[float]:
	""" Get the learning rates of an optimizer. """
	return [group["lr"] for group in optim.param_groups]


def get_lr(optim: Optimizer, idx: int = 0) -> float:
	""" Get the learning rate of an optimizer. """
	return get_lrs(optim)[idx]


def set_lr(optim: Optimizer, new_lr: float):
	""" Set the learning rate of an optimizer. """
	for group in optim.param_groups:
		group["lr"] = new_lr


def get_nb_parameters(model: Module) -> int:
	"""
		Return the number of parameters in a model.
		:param model: Pytorch Module to check.
		:returns: The number of parameters.
	"""
	return sum(p.numel() for p in model.parameters())


def get_nb_trainable_parameters(model: Module) -> int:
	"""
		Return the number of trainable parameters in a model.
		:param model: Pytorch Module.
		:returns: The number of trainable parameters.
	"""
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_reduction_from_name(name: str) -> Callable[[Tensor], Tensor]:
	"""
		Return the reduction function with a name. Available function are 'sum' and 'mean'.
	"""
	if name in ["mean", "batchmean"]:
		return torch.mean
	elif name in ["sum"]:
		return torch.sum
	elif name in ["none"]:
		return lambda x: x
	else:
		raise RuntimeError("Unknown reduction \"{:s}\". Must be in {:s}.".format(name, str(["mean", "sum", "none"])))


def mish(x: Tensor) -> Tensor:
	return x * torch.tanh(softplus(x))


class Mish(Module):
	"""
		Mish class for apply mish function.
	"""
	def __init__(self):
		super().__init__()

	def forward(self, x: Tensor) -> Tensor:
		return mish(x)


class Entropy(Module):
	def __init__(
		self,
		reduction: str = "batchmean",
		dim: int = 1,
		epsilon: float = DEFAULT_EPSILON,
		base: Optional[float] = None,
		log_input: bool = False,
	):
		"""
			Compute the entropy of a distribution.

			:param reduction: The reduction used between batch entropies.
			:param dim: The dimension to apply the sum in entropy formula.
			:param epsilon: The epsilon precision to use. Must be a small positive float.
			:param base: The log-base used. If None, use the natural logarithm (i.e. base = torch.exp(1)).
		"""
		super().__init__()
		self.reduce_fn = get_reduction_from_name(reduction)
		self.dim = dim
		self.epsilon = epsilon
		self.log_input = log_input

		if base is None:
			self.log_func = torch.log
		else:
			log_base = torch.log(torch.scalar_tensor(base))
			self.log_func = lambda x: torch.log(x) / log_base

	def forward(self, input_: Tensor, dim: Optional[int] = None) -> Tensor:
		if dim is None:
			dim = self.dim
		if not self.log_input:
			entropy = - torch.sum(input_ * self.log_func(input_ + self.epsilon), dim=dim)
		else:
			entropy = - torch.sum(torch.exp(input_) * input_, dim=dim)
		return self.reduce_fn(entropy)


class CrossEntropyWithVectors(Module):
	"""
		Compute Cross-Entropy between two distributions.
		Input and targets must be a batch of probabilities distributions of shape (batch_size, nb_classes) tensor.
	"""
	def __init__(self, reduction: str = "batchmean", dim: Optional[int] = 1, log_input: bool = False):
		super().__init__()
		self.reduce_fn = get_reduction_from_name(reduction)
		self.dim = dim
		self.log_input = log_input

	def forward(self, input_: Tensor, targets: Tensor, dim: Optional[int] = None) -> Tensor:
		"""
			Compute cross-entropy with targets.
			Input and target must be a (batch_size, nb_classes) tensor.
		"""
		if dim is None:
			dim = self.dim
		if not self.log_input:
			input_ = torch.log(input_)
		loss = -torch.sum(input_ * targets, dim=dim)
		return self.reduce_fn(loss)


class CrossEntropyOneHot(Module):
	def __init__(self, dim: int = 1):
		super().__init__()
		self.cross_entropy = CrossEntropyLoss()
		self.dim = dim

	def forward(self, logits: Tensor, target: Tensor) -> Tensor:
		target = target.argmax(dim=self.dim)
		return self.cross_entropy(logits, target)


class KLDivLossWithProbabilities(KLDivLoss):
	"""
		KL divergence with probabilities.
		The probabilities are transform to log scale internally.
	"""
	def __init__(self, reduction: str = "batchmean", epsilon: float = DEFAULT_EPSILON, log_input: bool = False, log_target: bool = False):
		super().__init__(reduction=reduction, log_target=True)
		self.epsilon = epsilon
		self.log_input = log_input
		self.log_target = log_target

	def forward(self, p: Tensor, q: Tensor) -> Tensor:
		if not self.log_input:
			p = torch.log(p + self.epsilon)
		if not self.log_target:
			q = torch.log(q + self.epsilon)
		return super().forward(input=p, target=q)


class JSDivLoss(Module):
	"""
		Jensen-Shannon Divergence loss.
		Use Entropy as backend.
	"""

	def __init__(self, reduction: str = "batchmean", dim: int = 1, epsilon: float = DEFAULT_EPSILON):
		super().__init__()
		self.entropy = Entropy(reduction, dim, epsilon)

	def forward(self, p: Tensor, q: Tensor) -> Tensor:
		a = self.entropy(0.5 * (p + q))
		b = 0.5 * (self.entropy(p) + self.entropy(q))
		return a - b


class JSDivLossWithLogits(Module):
	"""
		Jensen-Shannon Divergence loss with logits.
		Use KLDivLoss and LogSoftmax as backend.
	"""
	def __init__(self, reduction: str = "batchmean"):
		super().__init__()
		self.kl_div = KLDivLoss(reduction=reduction, log_target=True)
		self.log_softmax = LogSoftmax(dim=1)

	def forward(self, logits_p: Tensor, logits_q: Tensor):
		m = self.log_softmax(0.5 * (logits_p + logits_q))
		p = self.log_softmax(logits_p)
		q = self.log_softmax(logits_q)

		a = self.kl_div(p, m)
		b = self.kl_div(q, m)

		return 0.5 * (a + b)
