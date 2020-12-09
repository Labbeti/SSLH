"""
	WideResNet 28-2 class.
"""
import torch

from argparse import Namespace
from sslh.models.detail.wideresnet import ResNet, BasicBlock
from torch import Tensor, nn


class WideResNet28(ResNet):
	"""
		WideResNet-28 class.
	"""
	def __init__(self, num_classes: int, width: int = 2):
		super().__init__(layers=[4, 4, 4], width=width, num_classes=num_classes)

	@staticmethod
	def from_args(args: Namespace) -> 'WideResNet28':
		return WideResNet28(num_classes=args.nb_classes)


class WideResNet28Rot(ResNet):
	"""
		WideResNet-28 class with rotation layer.
	"""
	def __init__(self, num_classes: int, rot_size: int, width: int = 2):
		super().__init__(layers=[4, 4, 4], width=width, num_classes=num_classes)
		self.fc_rot = nn.Linear(64 * width * BasicBlock.expansion, rot_size)

	@staticmethod
	def from_args(args: Namespace) -> 'WideResNet28Rot':
		return WideResNet28Rot(num_classes=args.nb_classes, rot_size=args.nb_classes_self_supervised)

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


class WideResNet28Spec(WideResNet28):
	def __init__(self, num_classes: int, width: int = 2):
		super().__init__(num_classes, width)

	@staticmethod
	def from_args(args: Namespace) -> 'WideResNet28Spec':
		return WideResNet28Spec(num_classes=args.nb_classes)

	def forward(self, x: Tensor) -> Tensor:
		# duplicate spectrogram as input of WRN
		x = x.view(-1, 1, *x.shape[1:])
		x = x.repeat(1, 3, 1, 1)
		return super().forward(x)


class WideResNet28RotSpec(WideResNet28Rot):
	def __init__(self, num_classes: int, rot_size: int, width: int = 2):
		super().__init__(num_classes, rot_size, width)

	@staticmethod
	def from_args(args: Namespace) -> 'WideResNet28RotSpec':
		return WideResNet28RotSpec(rot_size=args.nb_classes_self_supervised, num_classes=args.nb_classes)

	def forward(self, x: Tensor) -> Tensor:
		# duplicate spectrogram as input of WRN
		x = x.view(-1, 1, *x.shape[1:])
		x = x.repeat(1, 3, 1, 1)
		return super().forward(x)

	def forward_rot(self, x: Tensor) -> Tensor:
		# duplicate spectrogram as input of WRN
		x = x.view(-1, 1, *x.shape[1:])
		x = x.repeat(1, 3, 1, 1)
		return super().forward_rot(x)
