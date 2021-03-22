
from typing import Callable, List
from mlu.transforms import Occlusion, CutOutSpec, StretchPadCrop


def get_weak_augm_pool() -> List[Callable]:
	return [
		Occlusion(scales=(0.1, 0.5), p=0.5),
		CutOutSpec(width_scales=(0.1, 0.5), height_scales=(0.1, 0.5), fill_value=-80.0, p=0.5),
		StretchPadCrop(rates=(0.5, 1.5), align="random", p=0.5),
	]


def get_strong_augm_pool() -> List[Callable]:
	return [
		Occlusion(scales=0.75, p=1.0),
		CutOutSpec(width_scales=(0.5, 1.0), height_scales=(0.5, 1.0), fill_value=-80.0, p=1.0),
		StretchPadCrop(rates=(0.25, 1.75), align="random", p=1.0),
	]
