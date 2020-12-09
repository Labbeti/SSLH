import numpy as np

from augmentation_utils.augmentations import SpecAugmentation
from augmentation_utils.spec_augmentations import Noise
from sslh.utils.misc import random_rect

from torch import Tensor
from typing import Optional, Tuple


class Identity(SpecAugmentation):
	def __init__(self):
		super().__init__(1.0)

	def apply_helper(self, data):
		return data


class IdentitySpec(Identity):
	pass


class CutOutSpec(SpecAugmentation):
	def __init__(
		self,
		ratio: float = 1.0,
		rect_width_scale_range: Tuple[float, float] = (0.1, 0.5),
		rect_height_scale_range: Tuple[float, float] = (0.1, 0.5),
		fill_value: Optional[float] = None
	):
		super().__init__(ratio)
		self.value_range = (-80.0, 0.0)
		self.rect_width_scale_range = rect_width_scale_range
		self.rect_height_scale_range = rect_height_scale_range
		self.fill_value = fill_value if fill_value is not None else self.value_range[0]

	def apply_helper(self, data: (Tensor, np.ndarray)) -> (Tensor, np.ndarray):
		width, height = data.shape[0], data.shape[1]
		r_left, r_right, r_top, r_down = random_rect(width, height, self.rect_width_scale_range, self.rect_height_scale_range)

		data[r_left:r_right, r_top:r_down] = self.fill_value

		return data


class InversionSpec(SpecAugmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)
		self.value_range = (-80.0, 0.0)

	def apply_helper(self, data):
		return self.value_range[1] + self.value_range[0] - data


class NoiseSpec(Noise):
	pass


class RandSpecAugment(SpecAugmentation):
	def __init__(self, ratio: float):
		super().__init__(ratio)
		self.augments_list = []
		# TODO

	def apply_helper(self, data):
		# TODO
		return data
