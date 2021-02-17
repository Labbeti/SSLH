
from argparse import Namespace
from torch import nn
from torch.nn import Module, Sequential

from sslh.models.detail.layers import ConvMish, ConvPoolMish


class CNN03Mish(Module):
	def __init__(self, output_size: int = 10, dropout: float = 0.5, **kwargs):
		super().__init__()

		self.features = Sequential(
			ConvPoolMish(1, 24, 3, 1, 1, (4, 2), (4, 2)),
			ConvPoolMish(24, 48, 3, 1, 1, (4, 2), (4, 2)),
			ConvPoolMish(48, 72, 3, 1, 1, (2, 2), (2, 2)),
			ConvPoolMish(72, 72, 3, 1, 1, (2, 2), (2, 2)),
			ConvMish(72, 72, 3, 1, 1),
		)

		self.classifier = Sequential(
			nn.Flatten(),
			nn.Dropout(dropout),
			nn.Linear(720, output_size),
		)

	def forward(self, x):
		x = x.view(-1, 1, *x.shape[1:])

		x = self.features(x)
		x = self.classifier(x)

		return x


class CNN03MishRot(CNN03Mish):
	def __init__(self, output_size: int = 10, dropout: float = 0.5, rot_output_size: int = 4, **kwargs):
		super().__init__(output_size, dropout, **kwargs)

		self.classifier_rot = Sequential(
			nn.Flatten(),
			nn.Dropout(dropout),
			nn.Linear(720, rot_output_size),
		)

	def forward_rot(self, x):
		x = x.view(-1, 1, *x.shape[1:])

		x = self.features(x)
		x = self.classifier_rot(x)

		return x
