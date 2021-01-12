
import numpy as np

from augmentation_utils.signal_augmentations import SignalAugmentation
from augmentation_utils.spec_augmentations import SpecAugmentation, Noise

from mlu.transforms import Transform

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import pad
from typing import Callable, Union


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


def is_waveform_augment(callable_: Callable) -> bool:
	"""
		:param callable_: A callable object to test.
		:return: True if the object is a SignalAugmentation from augmentation utils or WaveformTransform from mlu.
	"""
	return isinstance(callable_, SignalAugmentation) or (
			isinstance(callable_, Transform) and callable_.is_waveform_transform())


def is_spectrogram_augment(callable_: Callable) -> bool:
	"""
		:param callable_: A callable object to test.
		:return: True if the object is a SpecAugmentation from augmentation utils or SpectrogramTransform from mlu.
	"""
	return isinstance(callable_, SpecAugmentation) or (
			isinstance(callable_, Transform) and callable_.is_spectrogram_transform())
