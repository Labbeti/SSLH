
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, RandomCrop
from typing import Callable, List
from mlu.transforms import RandAugment, CutOutImgPIL


def get_weak_augm_pool() -> List[Callable]:
	return [
		RandomHorizontalFlip(0.5),
		RandomVerticalFlip(0.25),
		RandomCrop((32, 32), padding=8),
	]


def get_strong_augm_pool() -> List[Callable]:
	return [
		RandAugment(nb_augm_apply=1, magnitude_policy="random", augm_pool=None, p=1.0),
		CutOutImgPIL(scales=(0.2, 0.5), fill_value=0, p=1.0),
	]
