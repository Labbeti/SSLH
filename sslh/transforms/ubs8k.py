
import torch

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from typing import Callable, Optional

from mlu.nn import OneHot, UnSqueeze
from mlu.transforms import Compose, ToTensor, Pad, Crop
from sslh.transforms.pools.audio import get_pool
from sslh.transforms.self_transforms.audio import get_self_transform_flips
from sslh.transforms.utils import compose_augment

N_CLASSES = 10


def get_transform_ubs8k(augment_name: str, n_mels: int = 64, hop_length: int = 512, n_fft: int = 2048) -> Callable:
	pool = get_pool(augment_name)

	# Spectrogram shape : (channels, freq, time) = (1, 64, 173)
	pad_length = 4  # (seconds), max length of UBS8K waveforms
	sample_rate = 22050
	target_length = sample_rate * pad_length

	transform_to_spec = Compose(
		Crop(target_length),
		Pad(target_length),
		MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels),
		AmplitudeToDB(),
	)
	pre_transform = Compose(
		ToTensor(dtype=torch.float),
	)
	post_transform = UnSqueeze(dim=0)

	augment = compose_augment(pool, transform_to_spec, pre_transform, post_transform)
	return augment


def get_target_transform_ubs8k(smooth: Optional[float] = None) -> Callable:
	return OneHot(N_CLASSES, smooth, dtype=torch.float)


def get_self_transform_ubs8k(**kwargs) -> Callable:
	return get_self_transform_flips()
