
from sslh.fixmatch.loss import FixMatchLoss
from sslh.utils.torch import CrossEntropyWithVectors

from typing import Callable


class UDALoss(FixMatchLoss):
	""" UDA criterion used for training in UDA method. """
	def __init__(
		self,
		criterion_s: Callable = CrossEntropyWithVectors(reduction="none"),
		criterion_u: Callable = CrossEntropyWithVectors(reduction="none"),
	):
		"""
			:param criterion_s: The criterion used for labeled loss component.
			:param criterion_u: The criterion used for unlabeled loss component. No reduction must be applied.
		"""
		super().__init__(criterion_s, criterion_u)
