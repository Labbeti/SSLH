
import librosa
import numpy as np
import torch

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
	def __init__(self, target_length: int, mode: str = "constant", value: float = 0.0):
		super().__init__()
		self.target_length = target_length
		self.mode = mode
		self.value = value

	def forward(self, data: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
		length = data.shape[-1]
		pad_fn = lambda x: pad(input=x, pad=[0, (self.target_length - length)], mode=self.mode, value=self.value)
		if isinstance(data, np.ndarray):
			return pad_fn(torch.from_numpy(data)).numpy()
		else:
			return pad_fn(data)


class MelSpectrogramLibrosa(Module):
	def __init__(self, sr: int, n_fft: int, hop_length: int, n_mels: int):
		super().__init__()
		self.sr = sr
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.n_mels = n_mels
		self.fmin = 0
		self.fmax = sr // 2

	def forward(self, waveform: np.ndarray) -> np.ndarray:
		return librosa.feature.melspectrogram(
			waveform,
			sr=self.sr,
			n_fft=self.n_fft,
			hop_length=self.hop_length,
			n_mels=self.n_mels,
			fmin=self.fmin,
			fmax=self.fmax,
		)


class PowerToDbLibrosa(Module):
	def __init__(self):
		super().__init__()
		self.ref = np.max

	def forward(self, spectrogram: np.ndarray) -> np.ndarray:
		return librosa.power_to_db(spectrogram, ref=self.ref)


def is_waveform_augment(callable_: Callable) -> bool:
	"""
		:param callable_: A callable object to test.
		:return: True if the object is a SignalAugmentation from augmentation utils or WaveformTransform from mlu.
	"""
	return isinstance(callable_, SignalAugmentation) or (
		isinstance(callable_, Transform) and callable_.is_waveform_transform()
	)


def is_spectrogram_augment(callable_: Callable) -> bool:
	"""
		:param callable_: A callable object to test.
		:return: True if the object is a SpecAugmentation from augmentation utils or SpectrogramTransform from mlu.
	"""
	return isinstance(callable_, SpecAugmentation) or (
		isinstance(callable_, Transform) and callable_.is_spectrogram_transform()
	)
