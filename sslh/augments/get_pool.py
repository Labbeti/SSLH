
from argparse import Namespace
from augmentation_utils.signal_augmentations import Occlusion
from augmentation_utils.spec_augmentations import HorizontalFlip, RandomTimeDropout, RandomFreqDropout
from augmentation_utils.spec_augmentations import Noise as NoiseSpec

from mlu.transforms import Identity, RandomChoice
from mlu.transforms.image import RandAugment, CutOutImgPIL
from mlu.transforms.image.pools import RAND_AUGMENT_POOL_1, RAND_AUGMENT_POOL_2
from mlu.transforms.waveform import StretchPadCrop
from mlu.transforms.spectrogram import CutOutSpec

from sslh.augments.utils import is_waveform_augment

from torchvision.transforms import Compose, RandomCrop, RandomVerticalFlip, RandomHorizontalFlip, RandomAffine
from typing import Callable, List, Optional


def get_transform(name: Optional[str], args: Optional[Namespace], data_type: str, transform_base: Optional[Callable]) -> Callable:
	pool = get_pool(name, args, data_type)

	if data_type == "image":
		pool = add_postprocess_for_pool(pool, transform_base)
	elif data_type == "audio":
		pool = add_transform_to_spec_for_pool(pool, transform_base)
	else:
		raise RuntimeError(f"Unknown data type {data_type}.")

	transform = RandomChoice(*pool)
	return transform


def get_pool(name: Optional[str], args: Optional[Namespace], data_type: str) -> List[Callable]:
	if data_type == "image":
		return get_pool_img_with_name(name, args)
	elif data_type == "audio":
		return get_pool_audio_with_name(name, args)
	else:
		raise RuntimeError(f"Unknown data type {data_type}.")


def get_pool_img_with_name(name: Optional[str], args: Optional[Namespace]) -> List[Callable]:
	name = str(name).lower()

	if name in ["none"]:
		return [Identity()]

	elif name in ["weak1"]:
		ratio_augm_weak = 0.5
		return [
			HorizontalFlip(ratio_augm_weak),
			CutOutImgPIL(scales=(0.1, 0.1), fill_value=0, p=ratio_augm_weak),
		]

	elif name in ["strong", "strong1"]:
		ratio_augm_strong = 1.0
		nb_choices = args.ra_nb_choices if args is not None else 1

		if args.ra_pool is None:
			augm_pool = None
		elif args.ra_pool == "pool1":
			augm_pool = RAND_AUGMENT_POOL_1
		elif args.ra_pool == "pool2":
			augm_pool = RAND_AUGMENT_POOL_2
		else:
			raise RuntimeError(f"Invalid RA pool \"{args.ra_pool}\".")

		return [
			RandAugment(nb_augm_apply=nb_choices, magnitude_policy="random", p=ratio_augm_strong, augm_pool=augm_pool),
			CutOutImgPIL(scales=(0.2, 0.5), fill_value=0, p=ratio_augm_strong),
		]

	elif name in ["strong2"]:
		nb_choices = args.ra_nb_choices if args is not None else 1

		if args.ra_pool is None:
			augm_pool = None
		elif args.ra_pool == "pool1":
			augm_pool = RAND_AUGMENT_POOL_1
		elif args.ra_pool == "pool2":
			augm_pool = RAND_AUGMENT_POOL_2
		else:
			raise RuntimeError(f"Invalid RA pool \"{args.ra_pool}\".")

		return [
			RandAugment(nb_augm_apply=nb_choices, magnitude_policy="random", augm_pool=augm_pool),
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
			Compose([
				RandomHorizontalFlip(0.5),
				RandomCrop((32, 32), padding=8),
			])
		]

	elif name in ["strong3"]:
		return [
			Compose([
				RandAugment(nb_augm_apply=1, magnitude_policy="random", augm_pool=RAND_AUGMENT_POOL_2),
				CutOutImgPIL(scales=(0.0, 0.5)),
			])
		]

	elif name in ["strong4"]:
		return [
			Compose([
				RandAugment(nb_augm_apply=1, magnitude_policy="random", augm_pool=RAND_AUGMENT_POOL_2),
				CutOutImgPIL(scales=(0.5, 0.5)),
			])
		]

	else:
		raise RuntimeError(f"Unknown augment pool \"{name}\".")


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
			[augment, transform_to_spec] if is_waveform_augment(augment) else [transform_to_spec, augment]
		)
		for augment in pool
	]
