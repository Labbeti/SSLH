
from typing import Dict, Tuple

from mlu.metrics import Metric

from .ads import get_metrics_ads
from .cifar10 import get_metrics_cifar10
from .esc10 import get_metrics_esc10
from .gsc import get_metrics_gsc
from .pvc import get_metrics_pvc
from .ubs8k import get_metrics_ubs8k


def get_metrics(dataset_name: str) -> Tuple[Dict[str, Metric], Dict[str, Metric], Dict[str, Metric]]:
	"""
		Return the metrics used for a dataset.

		:param dataset_name: The name of the dataset. Can be one of: ('ADS', 'CIFAR10', 'ESC10', 'GSC', 'PVC', 'UBS8K').
		:return: A tuple (train metrics dict, validation metrics dict, validation stack metrics dict).
			A "stack" metric is a metric that must used all the predictions of the model as input (example: mAP metric).
			These metrics are computed only at the end of the training with the best model saved.
	"""
	dataset_name = dataset_name.upper()

	if dataset_name == "ADS":
		return get_metrics_ads()

	elif dataset_name == "CIFAR10":
		return get_metrics_cifar10()

	elif dataset_name == "ESC10":
		return get_metrics_esc10()

	elif dataset_name == "GSC":
		return get_metrics_gsc()

	elif dataset_name == "PVC":
		return get_metrics_pvc()

	elif dataset_name == "UBS8K":
		return get_metrics_ubs8k()

	else:
		raise RuntimeError(
			f"Unknown dataset name '{dataset_name}'. Must be one of {('ADS', 'CIFAR10', 'ESC10', 'GSC', 'PVC', 'UBS8K')}"
		)
