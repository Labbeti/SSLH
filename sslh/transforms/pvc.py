
from torch.nn import Sequential
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from typing import Callable, Optional

from mlu.nn import OneHot
from mlu.transforms import Pad
from sslh.transforms.pools.audio import get_pool
from sslh.transforms.self_transforms.audio import get_self_transform_flips
from sslh.transforms.utils import compose_augment

N_CLASSES = 5


def get_transform_pvc(augment_name: str, n_mels: int = 64, hop_length: int = 512, n_fft: int = 2048) -> Callable:
	pool = get_pool(augment_name)

	# Spectrogram shape : (channels, freq, time) = (1, 64, 94)
	waveform_length = 3  # seconds
	sample_rate = 16000
	target_length = sample_rate * waveform_length

	transform_to_spec = Sequential(
		Pad(target_length),
		MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels),
		AmplitudeToDB(),
	)
	pre_transform = None
	post_transform = None

	augment = compose_augment(pool, transform_to_spec, pre_transform, post_transform)
	return augment


def get_target_transform_pvc(smooth: Optional[float] = None) -> Optional[Callable]:
	return OneHot(N_CLASSES, smooth)


def get_self_transform_pvc(**kwargs) -> Callable:
	return get_self_transform_flips()
