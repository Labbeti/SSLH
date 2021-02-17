"""
	WideResNet 28-2 class.
"""
import torch

from sslh.models.wideresnet import WideResNet, BasicBlock
from torch import Tensor, nn


class WideResNet28(WideResNet):
	"""
		WideResNet-28 class. Expects an input of shape (bsize, 1, nb_mels, time stamps).
	"""
	def __init__(self, num_classes: int, width: int = 2, num_input_channels: int = 3):
		super().__init__(layers=[4, 4, 4], width=width, num_classes=num_classes, num_input_channels=num_input_channels)


class WideResNet28Rot(WideResNet):
	"""
		WideResNet-28 class with rotation layer. Expects an input of shape (bsize, 1, nb_mels, time stamps).
	"""
	def __init__(self, num_classes: int, rot_size: int, width: int = 2, num_input_channels: int = 3):
		super().__init__(layers=[4, 4, 4], width=width, num_classes=num_classes, num_input_channels=num_input_channels)
		self.fc_rot = nn.Linear(64 * width * BasicBlock.expansion, rot_size)

	def forward_rot(self, x: Tensor) -> Tensor:
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc_rot(x)

		return x


class WideResNet28Repeat(WideResNet28):
	def __init__(self, num_classes: int, width: int = 2):
		super().__init__(num_classes, width)

	def forward(self, x: Tensor) -> Tensor:
		# duplicate spectrogram as input of WRN (bsize, mels, times) => (bsize, 3, mels, times)
		x = x.view(-1, 1, *x.shape[1:])
		x = x.repeat(1, 3, 1, 1)
		return super().forward(x)


class WideResNet28RotRepeat(WideResNet28Rot):
	def __init__(self, num_classes: int, rot_size: int, width: int = 2):
		super().__init__(num_classes, rot_size, width)

	def forward(self, x: Tensor) -> Tensor:
		# duplicate spectrogram as input of WRN (bsize, mels, times) => (bsize, 3, mels, times)
		x = x.view(-1, 1, *x.shape[1:])
		x = x.repeat(1, 3, 1, 1)
		return super().forward(x)

	def forward_rot(self, x: Tensor) -> Tensor:
		# duplicate spectrogram as input of WRN (bsize, mels, times) => (bsize, 3, mels, times)
		x = x.view(-1, 1, *x.shape[1:])
		x = x.repeat(1, 3, 1, 1)
		return super().forward_rot(x)
