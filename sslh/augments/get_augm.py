
from argparse import Namespace
from augmentation_utils.signal_augmentations import Occlusion, TimeStretch
from augmentation_utils.spec_augmentations import HorizontalFlip, RandomTimeDropout, RandomFreqDropout, VerticalFlip
from augmentation_utils.spec_augmentations import Noise as NoiseSpec

from sslh.augments.signal_augments import ResizePadCut
from sslh.augments.spec_augments import CutOutSpec
from sslh.augments.utils import Identity

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
			- ResizePadCut :
				args.ratio: float,
				args.resize_rate: Union[Tuple[float, float], float],
				args.resize_align: str
	"""
	name = str(name).lower()
	ratio = args.ratio if args is not None else 1.0

	if name in ["Identity".lower(), "None".lower()]:
		return Identity(ratio)

	elif name == "HorizontalFlip".lower():
		return HorizontalFlip(ratio)

	elif name == "VerticalFlip".lower():
		return VerticalFlip(ratio)

	elif name == "Occlusion".lower():
		max_size = args.occlusion_max_size if args is not None else 1.0
		return Occlusion(
			ratio, max_size=max_size
		)
	elif name == "CutOutSpec".lower():
		width_scale = args.cutout_width_scale if args is not None else (0.1, 0.5)
		height_scale = args.cutout_height_scale if args is not None else (0.1, 0.5)
		return CutOutSpec(
			ratio, rect_width_scale_range=width_scale, rect_height_scale_range=height_scale
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
	elif name == "ResizePadCut".lower():
		rate = args.resize_rate if args is not None else (0.9, 1.1)
		align = args.resize_align if args is not None else "left"
		return ResizePadCut(
			ratio, rate=rate, align=align
		)
	else:
		raise RuntimeError(f"Unknown augment \"{name}\".")


def get_augments_by_names(names: List[str], args: Optional[Namespace]) -> List[Callable]:
	""" Get a list of augments with names. See get_augment_by_name function for details. """
	return [get_augment_by_name(name, args) for name in names]
