
from argparse import Namespace
from augmentation_utils.augmentations import SignalAugmentation
from augmentation_utils.signal_augmentations import Occlusion
from augmentation_utils.spec_augmentations import HorizontalFlip, RandomTimeDropout, RandomFreqDropout

from sslh.augments.img_augments import CutOut, IdentityImg
from sslh.augments.rand_augment import RandAugment
from sslh.augments.signal_augments import ResizePadCut
from sslh.augments.spec_augments import NoiseSpec, CutOutSpec, IdentitySpec

from torchvision.transforms import Compose, RandomCrop, RandomVerticalFlip, RandomHorizontalFlip
from typing import Callable, List, Optional


def get_pool_audio_with_name(name: Optional[str], args: Optional[Namespace]) -> List[Callable]:
	name = str(name).lower()

	if name in ["none"]:
		return [IdentitySpec()]

	elif name in ["weak1"]:
		ratio = 0.5
		return [
			HorizontalFlip(ratio),
			Occlusion(ratio, max_size=1.0)
		]

	elif name in ["strong1"]:
		ratio = 1.0
		return [
			CutOutSpec(ratio, rect_width_scale_range=(0.1, 0.25), rect_height_scale_range=(0.1, 0.25)),
			RandomTimeDropout(ratio, dropout=0.01),
			RandomFreqDropout(ratio, dropout=0.01),
			NoiseSpec(ratio, snr=5.0),
		]

	elif name in ["weak", "weak2"]:
		ratio = 0.5
		return [
			Occlusion(ratio, max_size=0.25),
			CutOutSpec(ratio, rect_width_scale_range=(0.1, 0.5), rect_height_scale_range=(0.1, 0.5)),
			ResizePadCut(ratio, rate=(0.5, 1.5), align="random"),
		]

	elif name in ["strong", "strong2"]:
		ratio = 1.0
		return [
			Occlusion(ratio, max_size=0.75),
			CutOutSpec(ratio, rect_width_scale_range=(0.5, 1.0), rect_height_scale_range=(0.5, 1.0)),
			ResizePadCut(ratio, rate=(0.25, 1.75), align="random"),
		]

	else:
		raise RuntimeError(f"Unknown augment pool \"{name}\".")


def get_pool_img_with_name(name: Optional[str], args: Optional[Namespace]) -> List[Callable]:
	name = str(name).lower()

	if name in ["none"]:
		return [IdentityImg()]

	elif name in ["weak", "weak1"]:
		ratio_augm_weak = 0.5
		return [
			HorizontalFlip(ratio_augm_weak),
			CutOut(ratio_augm_weak, rect_width_scale_range=(0.1, 0.1), rect_height_scale_range=(0.1, 0.1), fill_value=0),
		]

	elif name in ["strong", "strong1"]:
		ratio_augm_strong = 1.0
		magnitude = args.ra_magnitude if args is not None else 2.0
		nb_choices = args.ra_nb_choices if args is not None else 1
		return [
			CutOut(ratio_augm_strong, rect_width_scale_range=(0.2, 0.5), rect_height_scale_range=(0.2, 0.5), fill_value=0),
			RandAugment(ratio=ratio_augm_strong, magnitude_m=magnitude, nb_choices_n=nb_choices),
		]

	elif name in ["weak2"]:
		return [
			RandomHorizontalFlip(0.5),
			RandomVerticalFlip(0.25),
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
		Compose([augment, transform_to_spec] if isinstance(augment, SignalAugmentation) else [transform_to_spec, augment])
		for augment in pool
	]
