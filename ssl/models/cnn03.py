
from argparse import Namespace

from ssl.models.detail.layers import ConvReLU, ConvPoolReLU

from torch import nn
from torch.nn import Module, Sequential


class CNN03(Module):
	def __init__(self, output_size: int = 10, dropout: float = 0.5, **kwargs):
		super().__init__()

		self.features = Sequential(
			ConvPoolReLU(1, 24, 3, 1, 1, (4, 2), (4, 2)),
			ConvPoolReLU(24, 48, 3, 1, 1, (4, 2), (4, 2)),
			ConvPoolReLU(48, 72, 3, 1, 1, (2, 2), (2, 2)),
			ConvPoolReLU(72, 72, 3, 1, 1, (2, 2), (2, 2)),
			ConvReLU(72, 72, 3, 1, 1),
		)

		self.classifier = Sequential(
			nn.Flatten(),
			nn.Dropout(dropout),
			nn.Linear(720, output_size),
		)

	@staticmethod
	def from_args(args: Namespace) -> 'CNN03':
		return CNN03(
			output_size=args.nb_classes,
			dropout=args.dropout,
		)

	def forward(self, x):
		x = x.view(-1, 1, *x.shape[1:])

		x = self.features(x)
		x = self.classifier(x)

		return x


class CNN03Rot(CNN03):
	def __init__(self, output_size: int = 10, dropout: float = 0.5, rot_output_size: int = 4, **kwargs):
		super().__init__(output_size, dropout, **kwargs)

		self.classifier_rot = Sequential(
			nn.Flatten(),
			nn.Dropout(dropout),
			nn.Linear(720, rot_output_size),
		)

	@staticmethod
	def from_args(args: Namespace) -> 'CNN03Rot':
		return CNN03Rot(
			output_size=args.nb_classes,
			dropout=args.dropout,
			rot_output_size=args.nb_classes_self_supervised,
		)

	def forward_rot(self, x):
		x = x.view(-1, 1, *x.shape[1:])

		x = self.features(x)
		x = self.classifier_rot(x)

		return x
