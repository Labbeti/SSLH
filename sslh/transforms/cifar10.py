
from torchvision.transforms import ToTensor, Normalize
from typing import Callable, Optional

from mlu.nn import OneHot
from mlu.transforms import Compose
from sslh.transforms.pools.image import get_weak_augm_pool, get_strong_augm_pool
from sslh.transforms.self_transforms.image import get_self_transform_rotations
from sslh.transforms.utils import compose_augment


def get_transform_cifar10(transform_name: str) -> Callable:
	if transform_name == "weak":
		pool = get_weak_augm_pool()
	elif transform_name == "strong":
		pool = get_strong_augm_pool()
	elif transform_name == "identity":
		pool = []
	else:
		raise RuntimeError(f"Unknown transform name '{transform_name}'.")

	augment = compose_augment(pool, None, get_pre_transform_cifar10(), get_post_transform_cifar10())
	return augment


def get_pre_transform_cifar10() -> Optional[Callable]:
	return None


def get_post_transform_cifar10() -> Optional[Callable]:
	# Add postprocessing after each augmentation (shapes : [32, 32, 3] -> [3, 32, 32])
	return Compose(
		ToTensor(),
		Normalize(
			mean=(0.4914009, 0.48215896, 0.4465308),
			std=(0.24703279, 0.24348423, 0.26158753)
		),
	)


def get_target_transform_cifar10(smooth: Optional[float] = None) -> Callable:
	return OneHot(10, smooth)


def get_self_transform_cifar10() -> Callable:
	return get_self_transform_rotations()
