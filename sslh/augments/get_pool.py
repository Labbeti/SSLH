
from argparse import Namespace
from augmentation_utils.signal_augmentations import Occlusion
from augmentation_utils.spec_augmentations import HorizontalFlip, RandomTimeDropout, RandomFreqDropout
from augmentation_utils.spec_augmentations import Noise as NoiseSpec

from mlu.transforms import Identity, RandomChoice, Compose
from mlu.transforms.image import RandAugment, CutOutImgPIL
from mlu.transforms.image.ra_pools import RAND_AUGMENT_POOL_1, RAND_AUGMENT_POOL_2
from mlu.transforms.waveform import StretchPadCrop
from mlu.transforms.spectrogram import CutOutSpec

from sslh.augments.utils import is_waveform_augment
from sslh.datasets.base import DatasetBuilder

from torchvision.transforms import RandomCrop, RandomVerticalFlip, RandomHorizontalFlip, RandomAffine
from typing import Callable, List, Optional


def get_transform(
	pool_name: Optional[str],
	args: Optional[Namespace],
	builder: DatasetBuilder,
) -> Callable:
	"""
		Return the transform pool corresponding to a pool name.

		:param pool_name: The name of the pool. If None, the identity transform will be used.
		:param args: The optional argparse arguments.
		:param builder: The dataset builder for this transform.
		:return: The transform created with the pool name and the base transform.
	"""
	data_type = builder.get_data_type()
	transform_pre_process = builder.get_pre_transform()
	transform_post_process = builder.get_post_transform()
	transform_spec = builder.get_spec_transform()

	if data_type == "image":
		pool = get_pool_img_with_name(pool_name, args)
	elif data_type == "audio":
		pool = get_pool_audio_with_name(pool_name, args)
	else:
		raise RuntimeError(f"Unknown data type {data_type}.")

	pool_new = []
	for augm in pool:
		transforms = []

		if transform_pre_process is not None:
			transforms.append(transform_pre_process)

		if augm is not None:
			if transform_spec is not None:
				# Add transform to spectrogram before or after each augment depending on his internal type.
				if is_waveform_augment(augm):
					transforms.append(augm)
					transforms.append(transform_spec)
				else:
					transforms.append(transform_spec)
					transforms.append(augm)
			else:
				transforms.append(augm)
		elif transform_spec is not None:
			transforms.append(transform_spec)

		if transform_post_process is not None:
			transforms.append(transform_post_process)

		if len(transforms) == 0:
			raise RuntimeError("Found an empty list of processes.")
		elif len(transforms) == 1:
			pool_new.append(transforms[0])
		else:
			pool_new.append(Compose(*transforms))

	if len(pool) == 0:
		raise RuntimeError(f"Found empty pool '{pool_name}'.")
	elif len(pool_new) == 1:
		return pool_new[0]
	else:
		transform = RandomChoice(*pool_new)
		return transform


def get_pool_img_with_name(name: Optional[str], args: Optional[Namespace]) -> List[Callable]:
	"""
		:param name: The name of the image augments pool.
		:param args: The argparse arguments used for building the pool.
		:return: The list of augments of the pool.
	"""
	name = str(name).lower()

	if name in ["none", "identity"]:
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
			raise RuntimeError(f"Invalid RA pool '{args.ra_pool}'.")

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
			raise RuntimeError(f"Invalid RA pool '{args.ra_pool}'.")

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
			Compose(
				RandomAffine(0, translate=(1/16, 1/16)),
				RandomHorizontalFlip(),
			)
		]

	elif name in ["weak4"]:
		return [
			Compose(
				RandomHorizontalFlip(0.5),
				RandomCrop((32, 32), padding=8),
			)
		]

	elif name in ["strong3"]:
		return [
			Compose(
				RandAugment(nb_augm_apply=1, magnitude_policy="random", augm_pool=RAND_AUGMENT_POOL_2),
				CutOutImgPIL(scales=(0.0, 0.5)),
			)
		]

	elif name in ["strong4"]:
		return [
			Compose(
				RandAugment(nb_augm_apply=1, magnitude_policy="random", augm_pool=RAND_AUGMENT_POOL_2),
				CutOutImgPIL(scales=(0.5, 0.5)),
			)
		]

	else:
		raise RuntimeError(f"Unknown augment pool '{name}'.")


def get_pool_audio_with_name(name: Optional[str], args: Optional[Namespace]) -> List[Callable]:
	"""
		Notes :
		- Audio augmentations cannot be a Compose or RandomChoice because we cannot detect automatically if they are Waveform
		or Spectrogram augment when wrapped.
		- If all transform are MLU-Transforms, the problem is solved by the dynamic transform method "is_waveform_transform()".

		:param name: The name of the audio augments pool.
		:param args: The argparse arguments used for building the pool.
		:return: The list of augments of the pool.
	"""
	name = str(name).lower()

	if name in ["none", "identity"]:
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
			CutOutSpec(width_scales=(0.1, 0.25), height_scales=(0.1, 0.25), fill_value=-80.0, p=ratio),
			RandomTimeDropout(ratio, dropout=0.01),
			RandomFreqDropout(ratio, dropout=0.01),
			NoiseSpec(ratio, snr=5.0),
		]

	elif name in ["weak", "weak2"]:
		ratio = 0.5
		return [
			Occlusion(ratio, max_size=0.25),
			CutOutSpec(width_scales=(0.1, 0.5), height_scales=(0.1, 0.5), fill_value=-80.0, p=ratio),
			StretchPadCrop(rates=(0.5, 1.5), align="random", p=ratio),
		]

	elif name in ["strong", "strong2"]:
		ratio = 1.0
		return [
			Occlusion(ratio, max_size=0.75),
			CutOutSpec(width_scales=(0.5, 1.0), height_scales=(0.5, 1.0), fill_value=-80.0, p=ratio),
			StretchPadCrop(rates=(0.25, 1.75), align="random", p=ratio),
		]

	else:
		raise RuntimeError(f"Unknown augment pool '{name}'.")
