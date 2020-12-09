
import numpy as np
import torch

from augmentation_utils.augmentations import Augmentation
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import pad
from typing import Any, Callable, Union


class ToNumpy(Callable):
	def __call__(self, x: Any) -> np.ndarray:
		if isinstance(x, np.ndarray):
			return x
		elif isinstance(x, Tensor):
			return x.numpy()
		else:
			return np.asarray(x)


class Identity(Augmentation):
	def __init__(self, ratio: float = 1.0):
		super().__init__(ratio)

	def apply_helper(self, data):
		return data


class Squeeze(Module):
	def forward(self, x: Tensor) -> Tensor:
		return x.squeeze()


class UnSqueeze(Module):
	def __init__(self, dim: int):
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor) -> Tensor:
		return x.unsqueeze(self.dim)


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


class PadAndCut1D(Module):
	def __init__(self, target_length: int, align: str = "left", fill_value: float = 0.0):
		"""
			:param target_length: The length of the signal output.
			:param align: Alignment to use for cropping or padding. Can be 'left', 'right', 'center' or 'random'.
			:param fill_value: The value to use for fill in padding.
		"""
		super().__init__()
		self.target_length = target_length
		self.align = align
		self.fill_value = fill_value

	def forward(self, signal: Tensor) -> Tensor:
		"""
			Expect a 1D tensor of shape (size).
		"""
		align = self.align

		if len(signal) < self.target_length:
			if align == "left":
				signal = pad_left(signal, self.target_length, self.fill_value)
			elif align == "right":
				signal = pad_right(signal, self.target_length, self.fill_value)
			elif align == "center":
				signal = pad_center(signal, self.target_length, self.fill_value)
			elif align == "random":
				signal = pad_random(signal, self.target_length, self.fill_value)

		elif len(signal) > self.target_length:
			if align == "left":
				signal = cut_left(signal, self.target_length)
			elif align == "right":
				signal = cut_right(signal, self.target_length)
			elif align == "center":
				signal = cut_center(signal, self.target_length)
			elif align == "random":
				signal = cut_random(signal, self.target_length)

		return signal

	def set_target_length(self, target_length: int):
		self.target_length = target_length

	def set_align(self, align: str):
		self.align = align

	def set_fill_value(self, fill_value: float):
		self.fill_value = fill_value


def pad_left(signal: Tensor, target_length: int, fill_value: float) -> Tensor:
	""" Align to left by adding zeros to right. """
	assert target_length > len(signal)
	missing = target_length - len(signal)
	return torch.cat((signal, torch.full((missing,), fill_value)))


def pad_right(signal: Tensor, target_length: int, fill_value: float) -> Tensor:
	""" Align to right by adding zeros to left. """
	assert target_length > len(signal)
	missing = target_length - len(signal)
	return torch.cat((torch.full((missing,), fill_value), signal))


def pad_center(signal: Tensor, target_length: int, fill_value: float) -> Tensor:
	""" Align to center by adding half of zeros to left and the other half to right. """
	assert target_length > len(signal)
	missing = target_length - len(signal)
	missing_left = missing // 2 + missing % 2
	missing_right = missing // 2
	return torch.cat((
		torch.full(size=(missing_left,), fill_value=fill_value),
		signal,
		torch.full(size=(missing_right,), fill_value=fill_value)
	))


def pad_random(signal: Tensor, target_length: int, fill_value: float) -> Tensor:
	""" Randomly add zeros to left and right for having the size of target_length. """
	assert target_length > len(signal)
	missing = target_length - len(signal)
	missing_left = torch.randint(low=0, high=missing, size=()).item()
	missing_right = missing - missing_left
	return torch.cat((
		torch.full(size=(missing_left,), fill_value=fill_value),
		signal,
		torch.full(size=(missing_right,), fill_value=fill_value)
	))


def cut_left(signal: Tensor, target_length: int) -> Tensor:
	""" Align to left by removing values from right. """
	assert len(signal) > target_length
	return signal[:target_length]


def cut_right(signal: Tensor, target_length: int) -> Tensor:
	""" Align to right by removing values from left. """
	assert len(signal) > target_length
	start = len(signal) - target_length
	return signal[start:]


def cut_center(signal: Tensor, target_length: int) -> Tensor:
	""" Align to center by removing half of the values in left and the other half in right. """
	assert len(signal) > target_length
	diff = len(signal) - target_length
	start_idx = diff // 2 + diff % 2
	end_idx = start_idx + target_length
	return signal[start_idx:end_idx]


def cut_random(signal: Tensor, target_length: int) -> Tensor:
	""" Randomly remove values in left and right. """
	assert len(signal) > target_length
	diff = len(signal) - target_length
	start_idx = torch.randint(low=0, high=diff, size=()).item()
	end_idx = start_idx + target_length
	return signal[start_idx:end_idx]


class TimeStretchNearest(Module):
	def __init__(self, orig_freq: int = 16000, new_freq: int = 16000):
		super().__init__()
		self.orig_freq = orig_freq
		self.new_freq = new_freq

	def forward(self, data: Tensor) -> Tensor:
		indexes_orig = torch.arange(start=0, end=len(data), step=self.orig_freq / self.new_freq).round().long().clamp(max=len(data)-1)
		new_data = data[indexes_orig]
		return new_data
