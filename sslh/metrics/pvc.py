
from torch.nn import Module
from typing import Dict, Tuple
from mlu.metrics import AveragePrecision, CategoricalAccuracy, UAR


def get_metrics_pvc() -> Tuple[Dict[str, Module], Dict[str, Module], Dict[str, Module]]:
	train_metrics = {
		'acc': CategoricalAccuracy(),
	}
	val_metrics = {
		'acc': CategoricalAccuracy(),
	}
	val_metrics_stack = {
		'acc': CategoricalAccuracy(),
		'mAP': AveragePrecision(),
		'uar': UAR(),
	}
	return train_metrics, val_metrics, val_metrics_stack
