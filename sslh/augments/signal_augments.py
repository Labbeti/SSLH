
import numpy as np
import torch

from augmentation_utils.augmentations import SignalAugmentation

from sslh.augments.utils import TimeStretchNearest, PadAndCut1D

from torch import Tensor
from torch.distributions.uniform import Uniform
from typing import Tuple, Union


class ResizePadCut(SignalAugmentation):
	def __init__(
		self,
		ratio: float,
		rate: Union[Tuple[float, float], float] = (0.9, 1.1),
		align: str = "center"
	):
		"""
			:param ratio: The probability to apply the augmentation.
			:param rate: The ratio of the signal used for resize.
			:param align: Alignment to use for cropping or padding. Can be 'left', 'right', 'center' or 'random'.
		"""
		super().__init__(ratio)
		self.rate = rate

		self.uniform = Uniform(low=rate[0], high=rate[1]) if not isinstance(rate, float) else None
		self.time_stretch = TimeStretchNearest()
		self.pad_cut_transform = PadAndCut1D(target_length=0, align=align)

	def apply_helper(self, data: Union[np.ndarray, Tensor]) -> np.ndarray:
		assert len(data.shape) == 1
		data_length = data.shape[-1]
		self.time_stretch.orig_freq = data_length
		self.time_stretch.new_freq = round(data_length * self._get_rate())
		self.pad_cut_transform.set_target_length(data_length)

		if isinstance(data, np.ndarray):
			data = torch.from_numpy(data)

		data = self.time_stretch(data)
		data = self.pad_cut_transform(data)

		return data.numpy()

	def _get_rate(self) -> float:
		if self.uniform is None:
			return self.rate
		else:
			return self.uniform.sample().item()
