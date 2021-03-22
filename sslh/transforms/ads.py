
import torch

from torch.nn import Sequential
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from typing import Callable, Optional

from mlu.nn import UnSqueeze
from mlu.transforms import Identity, ToTensor
from sslh.transforms.self_transforms.audio import get_self_transform_flips
from sslh.transforms.pools.audio import get_weak_augm_pool, get_strong_augm_pool
from sslh.transforms.utils import compose_augment


def get_transform_ads(transform_name: str) -> Callable:
	if transform_name == "weak":
		pool = get_weak_augm_pool()
	elif transform_name == "strong":
		pool = get_strong_augm_pool()
	elif transform_name == "identity":
		pool = []
	else:
		raise RuntimeError(f"Unknown transform name '{transform_name}'.")

	augment = compose_augment(pool, get_transform_to_spec_ads(), get_pre_transform_ads(), get_post_transform_ads())
	return augment


def get_pre_transform_ads() -> Optional[Callable]:
	return ToTensor(dtype=torch.float)


def get_post_transform_ads() -> Optional[Callable]:
	return UnSqueeze(dim=0)


def get_transform_to_spec_ads() -> Optional[Callable]:
	# Spectrogram of shape (64, 500)
	n_mels = 64
	n_time = 500
	sr = 32000
	n_fft = 2048
	hop_length = sr * 10 // n_time

	return Sequential(
		MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels),
		AmplitudeToDB(),
	)


def get_target_transform_ads() -> Callable:
	return Identity()


def get_self_transform_ads() -> Callable:
	return get_self_transform_flips()
