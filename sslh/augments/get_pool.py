
from argparse import Namespace
from augmentation_utils.augmentations import SignalAugmentation
from augmentation_utils.signal_augmentations import Occlusion
from augmentation_utils.spec_augmentations import HorizontalFlip, RandomTimeDropout, RandomFreqDropout
from augmentation_utils.spec_augmentations import Noise as NoiseSpec

from mlu.transforms import Identity, Transform
from mlu.transforms.image import RandAugment, CutOutImgPIL
from mlu.transforms.waveform import StretchPadCrop
from mlu.transforms.spectrogram import CutOutSpec

from torchvision.transforms import Compose, RandomCrop, RandomVerticalFlip, RandomHorizontalFlip, RandomAffine
from typing import Callable, List, Optional


def get_pool_audio_with_name(name: Optional[str], args: Optional[Namespace]) -> List[Callable]:
	name = str(name).lower()

	if name in ["none"]:
		return [Identity()]

	elif name in ["weak1"]:
		ratio = 0.5
		return [
			HorizontalFlip(ratio),
			Occlusion(ratio, max_size=1.0)
		]

	elif name in ["strong1"]:
		ratio = 1.0
		return [
			CutOutSpec(width_scale_range=(0.1, 0.25), height_scale_range=(0.1, 0.25), fill_value=-80.0, p=ratio),
			RandomTimeDropout(ratio, dropout=0.01),
			RandomFreqDropout(ratio, dropout=0.01),
			NoiseSpec(ratio, snr=5.0),
		]

	elif name in ["weak", "weak2"]:
		ratio = 0.5
		return [
			Occlusion(ratio, max_size=0.25),
			CutOutSpec(width_scale_range=(0.1, 0.5), height_scale_range=(0.1, 0.5), fill_value=-80.0, p=ratio),
			StretchPadCrop(rate=(0.5, 1.5), align="random", p=ratio),
		]

	elif name in ["strong", "strong2"]:
		ratio = 1.0
		return [
			Occlusion(ratio, max_size=0.75),
			CutOutSpec(width_scale_range=(0.5, 1.0), height_scale_range=(0.5, 1.0), fill_value=-80.0, p=ratio),
			StretchPadCrop(rate=(0.25, 1.75), align="random", p=ratio),
		]

	else:
		raise RuntimeError(f"Unknown augment pool \"{name}\".")


def get_pool_img_with_name(name: Optional[str], args: Optional[Namespace]) -> List[Callable]:
	name = str(name).lower()

	if name in ["none"]:
		return [Identity()]

	elif name in ["weak1"]:
		ratio_augm_weak = 0.5
		return [
			HorizontalFlip(ratio_augm_weak),
			CutOutImgPIL(width_scale_range=(0.1, 0.1), height_scale_range=(0.1, 0.1), fill_value=0, p=ratio_augm_weak),
		]

	elif name in ["strong", "strong1"]:
		ratio_augm_strong = 1.0
		nb_choices = args.ra_nb_choices if args is not None else 1
		return [
			RandAugment(nb_augm_apply=nb_choices, magnitude_policy="random", p=ratio_augm_strong),
			CutOutImgPIL(width_scale_range=(0.2, 0.5), height_scale_range=(0.2, 0.5), fill_value=0, p=ratio_augm_strong),
		]

	elif name in ["strong2"]:
		return [
			RandAugment(nb_augm_apply=1, magnitude_policy="random"),
		]

	elif name in ["weak", "weak2"]:
		return [
			RandomHorizontalFlip(0.5),
			RandomVerticalFlip(0.25),
			RandomCrop((32, 32), padding=8),
		]

	elif name in ["weak3"]:
		return [
			Compose([
				RandomAffine(0, translate=(1/16, 1/16)),
				RandomHorizontalFlip(),
			])
		]

	elif name in ["weak4"]:
		return [
			RandomHorizontalFlip(0.5),
			RandomCrop((32, 32), padding=8),
		]

	else:
		raise RuntimeError(f"Unknown augment pool \"{name}\".")


def add_preprocess_for_pool(pool: List[Callable], pre_process: Callable) -> List[Callable]:
	return [Compose([pre_process, augment]) for augment in pool]


def add_postprocess_for_pool(pool: List[Callable], post_process: Callable) -> List[Callable]:
	return [Compose([augment, post_process]) for augment in pool]


def add_process_for_pool(
	pool: List[Callable],
	pre_process: Optional[Callable] = None,
	post_process: Optional[Callable] = None
) -> List[Callable]:
	if pre_process is not None and post_process is not None:
		return [Compose([pre_process, augment, post_process]) for augment in pool]
	elif pre_process is not None:
		return [Compose([pre_process, augment]) for augment in pool]
	elif post_process is not None:
		return [Compose([augment, post_process]) for augment in pool]
	else:
		return pool


def add_transform_to_spec_for_pool(pool: List[Callable], transform_to_spec: Callable) -> List[Callable]:
	return [
		Compose(
			[augment, transform_to_spec]
			if isinstance(augment, SignalAugmentation) or (isinstance(augment, Transform) and augment.is_waveform_transform())
			else
			[transform_to_spec, augment]
		)
		for augment in pool
	]
