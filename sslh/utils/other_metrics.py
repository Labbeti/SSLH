
import torch

from metric_utils.metrics import CategoricalAccuracy, Metrics
from sslh.utils.torch import CrossEntropyWithVectors, Entropy

from abc import ABC
from torch import Tensor
from torch.nn import Module
from typing import Callable, Tuple, Optional, Union


class CategoricalAccuracyOnehot(CategoricalAccuracy):
	"""
		Just Categorical Accuracy with a binarization with argmax().
		It takes one-hot vectors as input.
	"""

	def __init__(self, dim: int, epsilon: float = 1e-10):
		super().__init__(epsilon)
		self.dim = dim

	def __call__(self, input_: Tensor, target: Tensor) -> Tensor:
		with torch.no_grad():
			y_pred = input_.argmax(dim=self.dim)
			y_true = target.argmax(dim=self.dim)
			return super().__call__(y_pred, y_true)


class FuncMetric(Metrics):
	def __init__(self, fn: Callable[[Tensor, Tensor], Tensor], reduce_fn: Callable = torch.mean):
		super().__init__()
		self.fn = fn
		self.reduce_fn = reduce_fn

	def __call__(self, input_: Tensor, target: Tensor) -> Tensor:
		super().__call__(input_, target)

		with torch.no_grad():
			self.value_ = self.fn(input_, target)
			self.value_ = self.reduce_fn(self.value_)
			self.accumulate_value += self.value_

			return self.accumulate_value / self.count


class MaxMetric(FuncMetric):
	def __init__(self, dim: int):
		super().__init__(lambda input_, target: input_.max(dim=dim)[0])


class CrossEntropyMetric(FuncMetric):
	def __init__(self, dim: int, reduction: str = "batchmean", log_input: bool = False):
		ce = CrossEntropyWithVectors(dim=dim, reduction=reduction, log_input=log_input)
		super().__init__(lambda input_, target: ce(input_, target))


class EntropyMetric(FuncMetric):
	def __init__(self, dim: int, log_input: bool = False):
		entropy = Entropy(dim=dim, log_input=log_input)
		super().__init__(lambda input_, target: entropy(input_))


class EqConfidenceMetric(Metrics):
	def __init__(self, confidence: float, dim: Union[int, Tuple[int, int]], epsilon: float = 1e-10):
		super().__init__(epsilon)
		self.confidence = confidence
		self.dim = dim

	def __call__(self, input_: Tensor, target: Tensor) -> Tensor:
		super().__call__(input_, target)

		with torch.no_grad():
			input_ = input_.gt(self.confidence).float()
			target = target.gt(self.confidence).float()

			self.value_ = input_.eq(target)
			if isinstance(self.dim, int):
				self.value_ = self.value_.all(dim=self.dim)
			else:
				for d in sorted(self.dim, reverse=True):
					self.value_ = self.value_.all(dim=d)
			self.value_ = self.value_.float().mean()
			self.accumulate_value += self.value_
			return self.accumulate_value / self.count


class BinaryConfidenceAccuracy(Metrics):
	def __init__(self, confidence: float, epsilon: float = 1e-10):
		Metrics.__init__(self, epsilon)
		self.confidence = confidence

	def __call__(self, input_: Tensor, target: Tensor) -> Tensor:
		super().__call__(input_, target)

		with torch.no_grad():
			input_ = input_.gt(self.confidence).float()
			target = target.gt(self.confidence).float()
			correct = input_.eq(target).float().sum()
			self.value_ = correct / torch.prod(torch.as_tensor(target.shape))

			self.accumulate_value += self.value_
			return self.accumulate_value / self.count


class ContinueMetric(Module, ABC):
	def reset(self):
		raise NotImplementedError("Abstract method")

	def add(self, item: Tensor):
		raise NotImplementedError("Abstract method")

	def get_current(self) -> Optional[Tensor]:
		raise NotImplementedError("Abstract method")


class ContinueMean(ContinueMetric):
	def __init__(self):
		super().__init__()
		self._sum = None
		self._counter = 0

	def reset(self):
		self._sum = None
		self._counter = 0

	def add(self, item: Tensor):
		if isinstance(item, float):
			item = torch.scalar_tensor(item)
		if self._sum is None:
			self._sum = item
			self._counter = 1
		else:
			self._sum += item
			self._counter += 1

	def get_current(self) -> Optional[Tensor]:
		return self.get_mean()

	def get_mean(self) -> Optional[Tensor]:
		if self._sum is not None:
			return self._sum / self._counter
		else:
			return None


class ContinueStd(ContinueMetric):
	def __init__(self):
		super().__init__()
		self._items = []

	def reset(self):
		self._items = []

	def add(self, item: Tensor):
		if isinstance(item, float):
			item = torch.scalar_tensor(item)
		self._items.append(item)

	def get_current(self) -> Optional[Tensor]:
		return self.get_std()

	def get_std(self) -> Optional[Tensor]:
		return torch.stack(self._items).std(unbiased=False) if len(self._items) > 0 else None
