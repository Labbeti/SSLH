
import torch

from torch import Tensor
from torch.nn import Linear
from torch.nn import functional as F

from sslh.models.mobilenet import MobileNetV1, MobileNetV2


class MobileNetV1Rot(MobileNetV1):
	def __init__(self, n_classes: int, rot_size: int):
		super().__init__(n_classes=n_classes)
		self.rot_size = rot_size
		features_output_size = 1024
		self.fc_rot = Linear(features_output_size, rot_size)

	def forward_rot(self, x: Tensor) -> Tensor:
		x = self.features(x)
		x = torch.mean(x, dim=3)

		(x1, _) = torch.max(x, dim=2)
		x2 = torch.mean(x, dim=2)
		x = x1 + x2
		x = F.dropout(x, p=0.5, training=self.training)
		x = F.relu_(self.fc1(x))

		rot_output = self.fc_audioset(x)
		return rot_output


class MobileNetV2Rot(MobileNetV2):
	def __init__(self, n_classes: int, rot_size: int):
		super().__init__(n_classes=n_classes)
		self.rot_size = rot_size
		features_output_size = 1024
		self.fc_rot = Linear(features_output_size, rot_size)

	def forward_rot(self, x: Tensor) -> Tensor:
		x = self.features(x)
		x = torch.mean(x, dim=3)

		(x1, _) = torch.max(x, dim=2)
		x2 = torch.mean(x, dim=2)
		x = x1 + x2
		x = F.relu_(self.fc1(x))

		rot_output = self.fc_rot(x)
		return rot_output
