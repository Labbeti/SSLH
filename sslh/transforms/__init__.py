
from typing import Callable

from .ads import get_transform_ads, get_target_transform_ads, get_self_transform_ads
from .cifar10 import get_transform_cifar10, get_target_transform_cifar10, get_self_transform_cifar10
from .esc10 import get_transform_esc10, get_target_transform_esc10, get_self_transform_esc10
from .gsc import get_transform_gsc, get_target_transform_gsc, get_self_transform_gsc
from .pvc import get_transform_pvc, get_target_transform_pvc, get_self_transform_pvc
from .ubs8k import get_transform_ubs8k, get_target_transform_ubs8k, get_self_transform_ubs8k


def get_transform(dataset_name: str, transform_name: str) -> Callable:
	dataset_name = dataset_name.upper()

	if dataset_name == "ADS":
		return get_transform_ads(transform_name)
	elif dataset_name == "CIFAR10":
		return get_transform_cifar10(transform_name)
	elif dataset_name == "ESC10":
		return get_transform_esc10(transform_name)
	elif dataset_name == "GSC":
		return get_transform_gsc(transform_name)
	elif dataset_name == "PVC":
		return get_transform_pvc(transform_name)
	elif dataset_name == "UBS8K":
		return get_transform_ubs8k(transform_name)
	else:
		raise RuntimeError(
			f"Unknown dataset name '{dataset_name}'. "
			f"Must be one of {('ADS', 'CIFAR10', 'ESC10', 'GSC', 'PVC', 'UBS8K')}"
		)


def get_target_transform(dataset_name: str) -> Callable:
	dataset_name = dataset_name.upper()

	if dataset_name == "ADS":
		return get_target_transform_ads()
	elif dataset_name == "CIFAR10":
		return get_target_transform_cifar10()
	elif dataset_name == "ESC10":
		return get_target_transform_esc10()
	elif dataset_name == "GSC":
		return get_target_transform_gsc()
	elif dataset_name == "PVC":
		return get_target_transform_pvc()
	elif dataset_name == "UBS8K":
		return get_target_transform_ubs8k()
	else:
		raise RuntimeError(
			f"Unknown dataset name '{dataset_name}'. "
			f"Must be one of {('ADS', 'CIFAR10', 'ESC10', 'GSC', 'PVC', 'UBS8K')}"
		)


def get_self_transform(dataset_name: str) -> Callable:
	dataset_name = dataset_name.upper()

	if dataset_name == "ADS":
		return get_self_transform_ads()
	elif dataset_name == "CIFAR10":
		return get_self_transform_cifar10()
	elif dataset_name == "ESC10":
		return get_self_transform_esc10()
	elif dataset_name == "GSC":
		return get_self_transform_gsc()
	elif dataset_name == "PVC":
		return get_self_transform_pvc()
	elif dataset_name == "UBS8K":
		return get_self_transform_ubs8k()
	else:
		raise RuntimeError(
			f"Unknown dataset name '{dataset_name}'. "
			f"Must be one of {('ADS', 'CIFAR10', 'ESC10', 'GSC', 'PVC', 'UBS8K')}"
		)
