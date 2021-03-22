
from torch.nn import Sequential
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from typing import Callable, Optional

from mlu.nn import OneHot
from sslh.transforms.pools.audio import get_weak_augm_pool, get_strong_augm_pool
from sslh.transforms.self_transforms.audio import get_self_transform_flips
from sslh.transforms.utils import compose_augment


def get_transform_esc10(transform_name: str) -> Callable:
	if transform_name == "weak":
		pool = get_weak_augm_pool()
	elif transform_name == "strong":
		pool = get_strong_augm_pool()
	elif transform_name == "identity":
		pool = []
	else:
		raise RuntimeError(f"Unknown transform name '{transform_name}'.")

	augment = compose_augment(pool, get_transform_to_spec_esc10(), get_pre_transform_esc10(), get_post_transform_esc10())
	return augment


def get_pre_transform_esc10() -> Optional[Callable]:
	return None


def get_post_transform_esc10() -> Optional[Callable]:
	return None


def get_transform_to_spec_esc10() -> Optional[Callable]:
	# Spectrogram of shape (64, ~86)
	sample_rate = 44100
	n_fft = 2048
	hop_length = 512
	n_mels = 64

	return Sequential(
		MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels),
		AmplitudeToDB(),
	)


def get_target_transform_esc10(smooth: Optional[float] = None) -> Callable:
	return OneHot(10, smooth)


def get_self_transform_esc10() -> Callable:
	return get_self_transform_flips()
