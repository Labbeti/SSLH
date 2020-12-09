import math
import torch

from argparse import Namespace

from augmentation_utils.img_augmentations import Transform
from augmentation_utils.spec_augmentations import HorizontalFlip, VerticalFlip

from torch import Tensor
from torch.nn.functional import one_hot
from typing import Callable, Optional


def get_transform_self_supervised_rotate(args: Optional[Namespace] = None) -> Callable:
	angles_available = torch.as_tensor([0.0, math.pi / 2.0, math.pi, -math.pi / 2.0])

	def rotate(x: Tensor, angle: float) -> Tensor:
		rotation = Transform(1.0, rotation=(angle, angle))
		return rotation(x)

	def transform(batch: Tensor) -> (Tensor, Tensor):
		"""
			batch of shape (bsize, 3, height, width)
		"""
		labels_r = torch.randint(low=0, high=len(angles_available), size=[len(batch)])
		angles = angles_available[labels_r]
		device = batch.device

		batch_r = torch.stack([rotate(x, ang) for x, ang in zip(batch, angles)]).to(device).float()
		labels_r = one_hot(labels_r, len(angles_available)).to(device).float()
		return batch_r, labels_r

	return transform


def get_transform_self_supervised_flips(args: Optional[Namespace] = None) -> Callable:
	transforms = [
		lambda x: x,
		lambda x: HorizontalFlip(1.0)(x),
		lambda x: VerticalFlip(1.0)(x),
		lambda x: HorizontalFlip(1.0)(VerticalFlip(1.0)(x)),
	]
	if args is not None and args.nb_classes_self_supervised != len(transforms):
		raise RuntimeError("Nb classes self-supervised and nb of transforms are not equal.")

	def transform(batch: Tensor):
		labels_r = torch.randint(low=0, high=len(transforms), size=[len(batch)])

		device = batch.device
		batch = batch.cpu().numpy()

		batch_r = torch.as_tensor([
			transforms[idx](x) for x, idx in zip(batch, labels_r)
		]).to(device)

		labels_r = one_hot(labels_r, len(transforms)).float().to(device)

		return batch_r, labels_r

	return transform
