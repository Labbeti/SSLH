
from argparse import Namespace
from augmentation_utils.signal_augmentations import Occlusion as OcclusionSR
from augmentation_utils.signal_augmentations import TimeStretch, PitchShiftRandom
from augmentation_utils.spec_augmentations import (
	HorizontalFlip, RandomTimeDropout, RandomFreqDropout, VerticalFlip
)

from mlu.transforms import Identity, Compose
from mlu.transforms.waveform import StretchPadCrop
from mlu.transforms.spectrogram import CutOutSpec

from sslh.augments.utils import NoiseSpec, is_waveform_augment
from sslh.datasets.base import DatasetBuilder

from typing import Callable, List, Optional


def get_augment_by_name(name: Optional[str], args: Optional[Namespace]) -> Callable:
	"""
		Build the augment with name (case insensitive).

		If args is None, return the identity augment.

		If args is not None, it must contains the argument for building the augmentation :
			- Identity :
				args.ratio: float
			- HorizontalFlip :
				args.ratio: float
			- VerticalFlip :
				args.ratio: float
			- Occlusion :
				args.ratio: float,
				args.occlusion_sampling_rate: int,
				args.occlusion_max_size: float
			- CutOutSpec :
				args.ratio: float,
				args.cutout_width_scale: Tuple[float, float],
				args.cutout_height_scale: Tuple[float, float]
			- RandomTimeDropout :
				args.ratio: float,
				args.random_time_dropout: float
			- RandomTimeDropout :
				args.ratio: float,
				args.random_freq_dropout: float
			- NoiseSpec :
				args.ratio: float,
				args.noise_snr: float
			- TimeStretch :
				args.ratio: float,
				args.time_stretch_rate: Tuple[float, float]
			- StretchPadCrop :
				args.ratio: float,
				args.stretchpadcrop_rate: Union[Tuple[float, float], float],
				args.stretchpadcrop_align: str
			- PitchShiftRandom :
				args.psr_sampling_rate: int,
				args.psr_steps: Tuple[int, int]
	"""
	name = str(name).lower()
	ratio = args.ratio if args is not None else 1.0

	if name in ["Identity".lower(), "None".lower()]:
		return Identity()

	elif name == "HorizontalFlip".lower():
		return HorizontalFlip(ratio)

	elif name == "VerticalFlip".lower():
		return VerticalFlip(ratio)

	elif name in ["Occlusion".lower(), "OcclusionSR".lower()]:
		sampling_rate = args.occlusion_sampling_rate if args is not None else 22050
		max_size = args.occlusion_max_size if args is not None else 1.0
		return OcclusionSR(
			ratio=ratio, sampling_rate=sampling_rate, max_size=max_size
		)
	elif name == "CutOutSpec".lower():
		width_scales = args.cutout_width_scale if args is not None else (0.1, 0.5)
		height_scales = args.cutout_height_scale if args is not None else (0.1, 0.5)
		return CutOutSpec(
			width_scales=width_scales, height_scales=height_scales, fill_value=-80.0, p=ratio
		)
	elif name == "RandomTimeDropout".lower():
		random_time_dropout = args.random_time_dropout if args is not None else 0.01
		return RandomTimeDropout(
			ratio, dropout=random_time_dropout
		)
	elif name == "RandomFreqDropout".lower():
		random_freq_dropout = args.random_freq_dropout if args is not None else 0.01
		return RandomFreqDropout(
			ratio, dropout=random_freq_dropout
		)
	elif name in ["NoiseSpec".lower(), "Noise".lower()]:
		noise_snr = args.noise_snr if args is not None else 10.0
		return NoiseSpec(
			ratio, snr=noise_snr
		)
	elif name == "TimeStretch".lower():
		rate = args.time_stretch_rate if args is not None else (0.9, 1.1)
		return TimeStretch(
			ratio, rate=rate
		)
	elif name in ["StretchPadCrop".lower()]:
		rate = args.stretchpadcrop_rate if args is not None else (0.9, 1.1)
		align = args.stretchpadcrop_align if args is not None else "left"
		return StretchPadCrop(
			rate=rate, align=align, p=ratio
		)
	elif name in ["PitchShiftRandom".lower(), "PSR".lower()]:
		sampling_rate = args.psr_sampling_rate if args is not None else 22050
		steps = args.psr_steps if args is not None else (-3, 3)
		return PitchShiftRandom(
			ratio=ratio, sampling_rate=sampling_rate, steps=steps
		)
	else:
		raise RuntimeError(f"Unknown augment '{name}'.")


def get_augments_by_names(names: List[str], args: Optional[Namespace]) -> List[Callable]:
	""" Get a list of augments with names. See get_augment_by_name function for details. """
	return [get_augment_by_name(name, args) for name in names]


def add_process_transform(
	transform: Optional[Callable],
	transform_pre_process: Optional[Callable],
	transform_post_process: Optional[Callable],
	transform_spec: Optional[Callable],
) -> Optional[Callable]:
	transforms = []
	if transform_pre_process is not None:
		transforms.append(transform_pre_process)

	if transform is not None:
		if transform_spec is not None:
			if is_waveform_augment(transform):
				transforms.append(transform)
				transforms.append(transform_spec)
			else:
				transforms.append(transform_spec)
				transforms.append(transform)
		else:
			transforms.append(transform)
	elif transform_spec is not None:
		transforms.append(transform_spec)

	if transform_post_process is not None:
		transforms.append(transform_post_process)

	if len(transforms) == 0:
		return None
	elif len(transforms) == 1:
		return transforms[0]
	else:
		return Compose(*transforms)


def add_builder_process_transform(
	transform: Optional[Callable],
	builder: DatasetBuilder,
) -> Optional[Callable]:
	return add_process_transform(
		transform, builder.get_pre_transform(), builder.get_post_transform(), builder.get_spec_transform()
	)
