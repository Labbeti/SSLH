
from mlu.transforms import Compose, Identity, RandomChoice, Transform
from typing import Callable, List, Optional


def compose_augment(
	pool: List[Callable],
	transform_to_spec: Optional[Callable],
	pre_transform: Optional[Callable],
	post_transform: Optional[Callable],
) -> Callable:
	"""
		Compose augment pool with optional transform to spectrogram, pre-transform and post-transform.

		Ex:

		>>> from mlu.nn import UnSqueeze
		>>> from mlu.transforms import ToTensor
		>>> from mlu.transforms.spectrogram import VerticalFlip, HorizontalFlip
		>>> compose_augment([VerticalFlip(), HorizontalFlip()], None, ToTensor(), UnSqueeze(dim=0))
		... Compose(
		...		ToTensor(),
		...		RandomChoice(
		...			VerticalFlip(),
		...			HorizontalFlip()
		...		),
		...		UnSqueeze(dim=0),
		... )

		:param pool: The list of possible augments to apply.
		:param transform_to_spec: The optional transformation to spectrogram.
		:param pre_transform: The pre-transform to apply before augment & spectrogram.
		:param post_transform: The post-transform to apply after augment & spectrogram.
		:return: The augment pool composed as a Callable object.
	"""
	pool = add_transform_to_spec_to_pool(pool, transform_to_spec)
	augment = random_choice_pool(pool)
	augment = add_pre_post_transforms(pre_transform, augment, post_transform)
	return augment


def add_transform_to_spec_to_pool(pool: List[Callable], transform_to_spec: Optional[Callable]) -> List[Callable]:
	if len(pool) == 0:
		if transform_to_spec is None:
			return []
		else:
			return [transform_to_spec]

	is_waveform_augment = lambda t: isinstance(t, Transform) and t.is_waveform_transform()

	pool_new = []
	for augm in pool:
		transforms = []

		if augm is not None:
			if transform_to_spec is not None:
				# Add transform to spectrogram before or after each augment depending on his internal type.
				if is_waveform_augment(augm):
					transforms.append(augm)
					transforms.append(transform_to_spec)
				else:
					transforms.append(transform_to_spec)
					transforms.append(augm)
			else:
				transforms.append(augm)
		elif transform_to_spec is not None:
			transforms.append(transform_to_spec)

		if len(transforms) == 0:
			raise RuntimeError("Found an empty list of processes.")
		elif len(transforms) == 1:
			pool_new.append(transforms[0])
		else:
			pool_new.append(Compose(*transforms))
	return pool_new


def random_choice_pool(pool: List[Optional[Callable]]) -> Optional[Callable]:
	pool = [transform for transform in pool if transform is not None]

	if len(pool) == 0:
		return None
	elif len(pool) == 1:
		return pool[0]
	else:
		return RandomChoice(*pool)


def add_pre_post_transforms(*transforms: Optional[Callable]) -> Callable:
	pool = [transform for transform in transforms if transform is not None]

	if len(pool) == 0:
		return Identity()
	elif len(pool) == 1:
		return pool[0]
	else:
		return Compose(*pool)
