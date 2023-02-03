#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union

import numpy as np
import torch
import torchvision as tv

from PIL import Image
from torch import nn, Tensor


# --- MODULES ---


class ToList(nn.Module):
    """
    Convert a pytorch tensor, numpy array or PIL image to python list.
    """

    def forward(self, x: Union[list, np.ndarray, Tensor, Image.Image]) -> list:
        return to_list(x)


class ToNumpy(nn.Module):
    def __init__(self, dtype: Optional[np.dtype] = None) -> None:
        """
        Convert a python list, pytorch tensor or PIL image to numpy array.

        :param dtype: The optional dtype of the numpy array. (default: None)
        """
        super().__init__()
        self.dtype = dtype

    def forward(self, x: Union[list, np.ndarray, Tensor, Image.Image]) -> np.ndarray:
        return to_numpy(x, self.dtype)


class ToTensor(nn.Module):
    def __init__(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        permute_tensor: bool = True,
        normalize_tensor: bool = True,
    ) -> None:
        """
        Convert a python list, numpy array or PIL image to pytorch tensor.

        :param dtype: The optional dtype of the pytorch tensor.
                (default: None)
        :param device: The optional device of the pytorch tensor.
                (default: None)
        :param permute_tensor: Permute dimensions (height, width, channel) to (channel, height, width) when converting
                from PIL image.
                (default: True)
        :param normalize_tensor: Normalize the tensor values from [0, 255] to [0.0, 1.0]. when converting from PIL image.
                (default: True)
        """
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.permute_tensor = permute_tensor
        self.normalize_tensor = normalize_tensor

    def forward(self, x: Union[list, np.ndarray, Tensor, Image.Image]) -> Tensor:
        return to_tensor(
            x, self.dtype, self.device, self.permute_tensor, self.normalize_tensor
        )


class ToPIL(nn.Module):
    def __init__(
        self,
        mode: Optional[str] = "RGB",
        permute_tensor: bool = False,
        denormalize_tensor: bool = True,
    ) -> None:
        """
        Convert a pytorch tensor, numpy array or python list to PIL image.

        :param mode: Define the type and depth of a pixel in the image. (default: 'RGB')
                See https://pillow.readthedocs.io/en/5.1.x/handbook/concepts.html#modes for details on PIL modes.
        :param permute_tensor: Permute dimensions (height, width, channel) to (channel, height, width) for PIL image.
                (default: False)
        :param denormalize_tensor: Normalize tensor values from [0.0, 1.0] to [0, 255].
                (default: True)
        """
        super().__init__()
        self.mode = mode
        self.permute_tensor = permute_tensor
        self.denormalize_tensor = denormalize_tensor

    def forward(self, x: Union[list, np.ndarray, Tensor, Image.Image]) -> Image.Image:
        return to_pil(x, self.mode, self.permute_tensor, self.denormalize_tensor)


# --- FUNCTIONS ---


def to_list(
    x: Union[list, np.ndarray, Tensor, Image.Image],
) -> list:

    if isinstance(x, list):
        return x
    elif isinstance(x, np.ndarray) or isinstance(x, Tensor):
        return x.tolist()
    elif isinstance(x, Image.Image):
        return np.asarray(x).tolist()
    else:
        return list(x)


def to_numpy(
    x: Union[list, np.ndarray, Tensor, Image.Image],
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    :param x: The list, numpy array, pytorch tensor or pillow image to convert.
    :param dtype: The optional dtype of the numpy array.
    """

    if isinstance(x, list) or isinstance(x, Image.Image):
        return np.asarray(x, dtype=dtype)
    elif isinstance(x, np.ndarray):
        return np.array(x, dtype=dtype)
    elif isinstance(x, Tensor):
        return np.array(x.cpu().numpy(), dtype=dtype)
    else:
        return np.asarray(x, dtype=dtype)


def to_tensor(
    x: Union[list, np.ndarray, Tensor, Image.Image],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = torch.device("cpu"),
    permute_tensor: bool = True,
    normalize_tensor: bool = True,
) -> Tensor:

    if isinstance(x, list):
        return torch.as_tensor(x, dtype=dtype, device=device)  # type: ignore
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x.copy()).to(dtype=dtype, device=device)
    elif isinstance(x, Tensor):
        return x.to(dtype).to(device)
    elif isinstance(x, Image.Image):
        x = to_tensor(to_numpy(x))
        if permute_tensor:
            # Permute dimensions (height, width, channel) to (channel, height, width).
            x = x.permute(2, 0, 1)
        if normalize_tensor:
            x = x / 255.0
        return torch.as_tensor(x, dtype=dtype, device=device)  # type: ignore
    else:
        return torch.as_tensor(x, dtype=dtype, device=device)  # type: ignore


def to_pil(
    x: Union[list, np.ndarray, Tensor, Image.Image],
    mode: Optional[str] = "RGB",
    permute_tensor: bool = False,
    denormalize_tensor: bool = True,
) -> Image.Image:

    if isinstance(x, list):
        return Image.fromarray(np.asarray(x), mode)
    elif isinstance(x, np.ndarray):
        to_pil_image = tv.transforms.ToPILImage(mode)
        return to_pil_image(x)
    elif isinstance(x, Tensor):
        if permute_tensor:
            # Permute dimensions (height, width, channel) to (channel, height, width).
            x = x.permute(2, 0, 1)
        if denormalize_tensor:
            x = x * 255
        to_pil_image = tv.transforms.ToPILImage(mode)
        return to_pil_image(x)
    elif isinstance(x, Image.Image):
        return x
    else:
        return Image.fromarray(x, mode)
