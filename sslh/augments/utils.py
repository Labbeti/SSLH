
import numpy as np

from augmentation_utils.spec_augmentations import Noise
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import pad
from typing import Union


class NoiseSpec(Noise):
	pass


class PadUpTo(Module):
	def __init__(self, target_length, mode: str = "constant", value: float = 0.0):
		super().__init__()
		self.target_length = target_length
		self.mode = mode
		self.value = value

	def forward(self, x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
		actual_length = x.shape[-1]
		x_pad = pad(input=x, pad=[0, (self.target_length - actual_length)], mode=self.mode, value=self.value)
		if isinstance(x, np.ndarray):
			return x_pad.numpy()
		else:
			return x_pad
