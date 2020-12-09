
from sslh.datasets.abc import DatasetInterface
from sslh.datasets.audioset import AudioSetInterface
from sslh.datasets.cifar10 import CIFAR10Interface
from sslh.datasets.esc10 import ESC10Interface
from sslh.datasets.esc50 import ESC50Interface
from sslh.datasets.gsc import GSCInterface
from sslh.datasets.gsc12 import GSC12Interface
from sslh.datasets.ubs8k import UBS8KInterface


INTERFACES = [
	AudioSetInterface,
	CIFAR10Interface,
	ESC10Interface,
	ESC50Interface,
	GSCInterface,
	GSC12Interface,
	UBS8KInterface,
]


def get_dataset_interface(name: str) -> DatasetInterface:
	for interface_cls in INTERFACES:
		interface = interface_cls()
		if interface.get_dataset_name().lower() == name.lower():
			return interface

	raise RuntimeError("Unknown dataset \"{:s}\".")
