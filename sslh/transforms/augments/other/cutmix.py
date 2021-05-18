
import math
import torch

from torch import Tensor
from torch.nn import Module
from torch.distributions.beta import Beta


class CutMix(Module):
	def __init__(self, alpha: float = 1.0):
		super().__init__()

		self._distribution = Beta(alpha, alpha)
		self._lambda = torch.zeros(1)

	def forward(self, batch_a: Tensor, batch_b: Tensor, label_a: Tensor, label_b: Tensor) -> (Tensor, Tensor):
		"""
			Apply CutMix strategy.

			:param batch_a: Shape (bsize, n_channels, width, height)
			:param batch_b: Shape (bsize, n_channels, width, height)
			:param label_a: Shape (bsize, n_classes)
			:param label_b: Shape (bsize, n_classes)
			:return: The image mixed and his label respectively of shapes (bsize, n_channels, width, height) and (bsize, n_classes).
		"""
		assert batch_a.shape == batch_b.shape and label_a.shape == label_b.shape

		self._lambda = self._distribution.sample()
		width, height = batch_a.shape[2:4]

		bbx1, bby1, bbx2, bby2 = self.gen_rand_bbox(width, height, self._lambda)
		batch_mix = batch_a.clone()
		batch_mix[:, :, bbx1:bbx2, bby1:bby2] = batch_b[:, :, bbx1:bbx2, bby1:bby2]

		# adjust lambda to exactly match pixel ratio
		self._lambda = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (width * height))
		label_mix = label_a * self._lambda + label_b * (1.0 - self._lambda)

		return batch_mix, label_mix

	def gen_rand_bbox(self, width: int, height: int, lambda_: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
		cut_ratio = math.sqrt(1.0 - lambda_)

		# rw, rh, rx, ry in paper
		cut_width = torch.scalar_tensor(math.floor(width * cut_ratio))
		cut_height = torch.scalar_tensor(math.floor(height * cut_ratio))
		center_x = torch.randint(low=0, high=width, size=())
		center_y = torch.randint(low=0, high=height, size=())

		bbx1 = torch.clip(center_x - cut_width // 2, min=0, max=width)
		bbx2 = torch.clip(center_x + cut_width // 2, min=0, max=width)

		bby1 = torch.clip(center_y - cut_height // 2, min=0, max=height)
		bby2 = torch.clip(center_y + cut_height // 2, min=0, max=height)

		return bbx1, bby1, bbx2, bby2

	def get_last_lambda(self) -> float:
		return self._lambda.item()
