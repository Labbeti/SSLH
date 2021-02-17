
from sslh.datasets.base import DatasetBuilder
from sslh.datasets.audioset import AudioSetBuilder
from sslh.datasets.cifar10 import CIFAR10Builder
from sslh.datasets.esc10 import ESC10Builder
from sslh.datasets.esc50 import ESC50Builder
from sslh.datasets.gsc import GSCBuilder
from sslh.datasets.gsc12 import GSC12Builder
from sslh.datasets.ubs8k import UBS8KBuilder


BUILDERS = [
	AudioSetBuilder,
	CIFAR10Builder,
	ESC10Builder,
	ESC50Builder,
	GSCBuilder,
	GSC12Builder,
	UBS8KBuilder,
]


def get_dataset_builder(name: str) -> DatasetBuilder:
	name = name.lower()
	for builder_type in BUILDERS:
		builder = builder_type()
		if builder.get_dataset_name().lower() == name:
			return builder

	raise RuntimeError(f"Unknown dataset '{name}'.")
