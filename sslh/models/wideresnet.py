"""
	WideResNet 28 based on pytorch implementation of ResNet.
"""

import torch
import torch.nn as nn

from torch.nn import Module
from typing import Callable, List, Optional, Tuple, Type


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> Module:
	"""3x3 convolution with padding"""
	return nn.Conv2d(
		in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> Module:
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(
		self,
		in_planes: int,
		planes: int,
		norm_layer: Callable[[int], Module],
		stride: int = 1,
		down_sample: Optional[Module] = None,
		groups: int = 1,
		base_width: int = 64,
		dilation: int = 1,
	):
		super(BasicBlock, self).__init__()

		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(in_planes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.down_sample = down_sample
		self.stride = stride

		self.expansion = 2

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.down_sample is not None:
			identity = self.down_sample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(Module):
	def __init__(self):
		super().__init__()
		raise NotImplementedError


class WideResNet(nn.Module):
	def __init__(
		self,
		layers: List[int],
		n_classes: int,
		width: int = 2,
		n_input_channels: int = 3,
		zero_init_residual: bool = False,
		groups: int = 1,
		width_per_group: int = 16,
		replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
		norm_layer: Optional[Type[Module]] = None
	):
		nn.Module.__init__(self)

		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		block = BasicBlock
		self.in_planes = 16 * width
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError(
				'replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation)
			)
		self.groups = groups
		self.base_width = width_per_group
		self.conv1 = nn.Conv2d(
			n_input_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = norm_layer(self.in_planes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(
			block, 16*width, layers[0])
		self.layer2 = self._make_layer(
			block, 32*width, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(
			block, 64*width, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(64 * width * block.expansion, n_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		down_sample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.in_planes != planes * block.expansion:
			down_sample = nn.Sequential(
				conv1x1(self.in_planes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers = [block(
			in_planes=self.in_planes,
			planes=planes,
			stride=stride,
			down_sample=down_sample,
			groups=self.groups,
			base_width=self.base_width,
			dilation=previous_dilation,
			norm_layer=norm_layer,
		)]

		self.in_planes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(
				in_planes=self.in_planes,
				planes=planes,
				groups=self.groups,
				base_width=self.base_width,
				dilation=self.dilation,
				norm_layer=norm_layer,
			))

		return nn.Sequential(*layers)

	def _forward_impl(self, x):
		# See note [TorchScript super()]
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)

		return x

	def forward(self, x):
		return self._forward_impl(x)
