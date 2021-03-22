
from torch.nn import Sequential
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from typing import Callable, Optional

from mlu.nn import OneHot
from mlu.transforms import PadAlignLeft
from sslh.transforms.pools.audio import get_weak_augm_pool, get_strong_augm_pool
from sslh.transforms.self_transforms.audio import get_self_transform_flips
from sslh.transforms.utils import compose_augment


def get_transform_pvc(transform_name: str) -> Callable:
	if transform_name == "weak":
		pool = get_weak_augm_pool()
	elif transform_name == "strong":
		pool = get_strong_augm_pool()
	elif transform_name == "identity":
		pool = []
	else:
		raise RuntimeError(f"Unknown transform name '{transform_name}'.")

	augment = compose_augment(pool, get_transform_to_spec_pvc(), get_pre_transform_pvc(), get_post_transform_pvc())
	return augment


def get_pre_transform_pvc() -> Optional[Callable]:
	return None


def get_post_transform_pvc() -> Optional[Callable]:
	return None


def get_transform_to_spec_pvc() -> Optional[Callable]:
	# Spec shape : (..., freq, time)
	waveform_length = 3  # second
	sr = 16000
	n_fft = 2048  # window size
	hop_length = 512
	n_mels = 64

	return Sequential(
		PadAlignLeft(target_length=sr * waveform_length, fill_value=0.0),
		MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels),
		AmplitudeToDB(),
	)


def get_target_transform_pvc(smooth: Optional[float] = None) -> Optional[Callable]:
	return OneHot(5, smooth)


def get_self_transform_pvc() -> Callable:
	return get_self_transform_flips()
