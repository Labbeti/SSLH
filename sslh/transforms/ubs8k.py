
import torch

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from typing import Callable, Optional

from mlu.nn import OneHot, UnSqueeze
from mlu.transforms import Compose, ToTensor, PadCrop
from sslh.transforms.pools.audio import get_weak_augm_pool, get_strong_augm_pool
from sslh.transforms.self_transforms.audio import get_self_transform_flips
from sslh.transforms.utils import compose_augment


def get_transform_ubs8k(transform_name: str) -> Callable:
	if transform_name == "weak":
		pool = get_weak_augm_pool()
	elif transform_name == "strong":
		pool = get_strong_augm_pool()
	elif transform_name == "identity":
		pool = []
	else:
		raise RuntimeError(f"Unknown transform name '{transform_name}'.")

	augment = compose_augment(pool, get_transform_to_spec_ubs8k(), get_pre_transform_ubs8k(), get_post_transform_ubs8k())
	return augment


def get_pre_transform_ubs8k() -> Optional[Callable]:
	pad_length = 4  # (seconds), max length of UBS8K waveforms
	sample_rate = 22050
	target_length = sample_rate * pad_length
	return Compose(
		ToTensor(dtype=torch.float),
		PadCrop(target_length=target_length, align="left"),
	)


def get_post_transform_ubs8k() -> Optional[Callable]:
	return UnSqueeze(dim=0)


def get_transform_to_spec_ubs8k() -> Optional[Callable]:
	sample_rate = 22050
	n_fft = 2048  # window size
	hop_length = 512
	n_mels = 64

	return Compose(
		MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels),
		AmplitudeToDB(),
	)


def get_target_transform_ubs8k(smooth: Optional[float] = None) -> Callable:
	return OneHot(10, smooth)


def get_self_transform_ubs8k() -> Callable:
	return get_self_transform_flips()
