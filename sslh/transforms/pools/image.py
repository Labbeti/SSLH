
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, RandomCrop
from typing import Callable, List, Tuple

from mlu.transforms import RandAugment, CutOutImgPIL
from mlu.transforms.image.ra_pools import RAND_AUGMENT_DEFAULT_POOL


def get_pool(name: str) -> List[Tuple[str, Callable]]:
	if name in ['weak']:
		pool = get_weak_augm_pool()
	elif name in ['strong']:
		pool = get_strong_augm_pool()
	elif name in ['identity']:
		pool = []
	else:
		raise RuntimeError(f'Unknown transform name "{name}".')
	return pool


def get_weak_augm_pool() -> List[Tuple[str, Callable]]:
	return [
		('image', RandomHorizontalFlip(0.5)),
		('image', RandomVerticalFlip(0.25)),
		('image', RandomCrop((32, 32), padding=8)),
	]


def get_strong_augm_pool() -> List[Tuple[str, Callable]]:
	return [
		('image', RandAugment(n_augm_apply=1, magnitude_policy='random', augm_pool=RAND_AUGMENT_DEFAULT_POOL, p=1.0)),
		('image', CutOutImgPIL(scales=(0.2, 0.5), fill_value=0, p=1.0)),
	]
