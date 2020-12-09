
from argparse import Namespace
from torch import nn, Tensor
from sslh.models.desed_baseline import WeakBaseline, WeakStrongBaseline


class WeakBaselineRot(WeakBaseline):
	def __init__(self, rot_output_size: int = 4):
		super().__init__()
		self.classifier_rot = nn.Sequential(
			nn.Flatten(),
			nn.Linear(1696, rot_output_size)
		)

	@staticmethod
	def from_args(args: Namespace) -> 'WeakBaselineRot':
		return WeakBaselineRot(
			rot_output_size=args.nb_classes_self_supervised,
		)

	def forward_rot(self, x: Tensor) -> Tensor:
		# Fox ReMixMatch
		x = x.view(-1, 1, *x.shape[1:])

		x = self.features(x)
		x = self.classifier_rot(x)

		return x


class WeakBaselineCount(WeakBaseline):
	def __init__(self, output_size: int = 10):
		super().__init__()
		self.classifier_count = nn.Sequential(
			nn.Flatten(),
			nn.Linear(1696, output_size + 1)
		)

	@staticmethod
	def from_args(args: Namespace) -> 'WeakBaselineCount':
		return WeakBaselineCount(
			output_size=args.nb_classes,
		)

	def forward_count(self, x: Tensor) -> Tensor:
		# For FixMatch V4 tag only
		x = x.view(-1, 1, *x.shape[1:])

		x = self.features(x)
		x = self.classifier_count(x)

		return x


class WeakStrongBaselineRot(WeakStrongBaseline):
	def __init__(self, rot_output_size: int = 4):
		super().__init__()

		self.classifier_rot = nn.Sequential(
			nn.Flatten(),
			nn.Linear(1696, rot_output_size)
		)

	@staticmethod
	def from_args(args: Namespace) -> 'WeakStrongBaselineRot':
		return WeakStrongBaselineRot(
			rot_output_size=args.nb_classes_self_supervised,
		)

	def forward_rot(self, x: Tensor) -> Tensor:
		# Fox ReMixMatch
		x = x.view(-1, 1, *x.shape[1:])

		x = self.features(x)
		x = self.classifier_rot(x)

		return x
