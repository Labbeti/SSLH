
import torch

from torch import Tensor
from torch.nn.functional import one_hot
from typing import Callable, Tuple

from mlu.transforms import Compose, Identity
from mlu.transforms.spectrogram import HorizontalFlip, VerticalFlip


def get_self_transform_flips() -> Callable:
	transforms = [
		Identity(),
		HorizontalFlip(),
		VerticalFlip(),
		Compose(HorizontalFlip(), VerticalFlip()),
	]

	def generate_flips(x: Tensor) -> Tuple[Tensor, Tensor]:
		bsize = len(x)
		class_idx = torch.randint(low=0, high=len(transforms), size=(bsize,))

		xr = torch.empty_like(x)
		for i, image in enumerate(x):
			transform = transforms[class_idx[i]]
			xr[i] = transform(image)

		yr = one_hot(class_idx, len(transforms)).to(device=x.device, dtype=x.dtype)
		return xr, yr

	return generate_flips
