
from argparse import Namespace
from sslh.models.detail.layers import ConvPoolReLU
from torch import nn


class UBS8KBaseline(nn.Module):
	def __init__(self, output_size: int = 10, dropout: float = 0.5, **kwargs):
		nn.Module.__init__(self)

		self.features = nn.Sequential(
			ConvPoolReLU(1, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.0),
			ConvPoolReLU(32, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.3),
			ConvPoolReLU(32, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.3),
			nn.Conv2d(32, 32, 1, 1, 0),
			nn.ReLU6(),
		)

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(dropout),
			nn.Linear(672, output_size)
		)

	def forward(self, x):
		x = x.view(-1, 1, *x.shape[1:])

		x = self.features(x)
		x = self.classifier(x)

		return x


class UBS8KBaselineRot(UBS8KBaseline):
	def __init__(self, output_size: int = 10, dropout: float = 0.5, rot_output_size: int = 4, **kwargs):
		super().__init__(output_size, dropout, **kwargs)
		self.classifier_rot = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(dropout),
			nn.Linear(672, rot_output_size)
		)

	def forward_rot(self, x):
		x = x.view(-1, 1, *x.shape[1:])

		x = self.features(x)
		x = self.classifier_rot(x)

		return x
