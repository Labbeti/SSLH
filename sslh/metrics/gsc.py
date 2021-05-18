
from torch.nn import Module
from typing import Dict, Tuple
from mlu.metrics import CategoricalAccuracy, MetricWrapper
from mlu.nn import CrossEntropyWithVectors


def get_metrics_gsc() -> Tuple[Dict[str, Module], Dict[str, Module], Dict[str, Module]]:
	train_metrics = {
		'acc': CategoricalAccuracy(),
	}
	val_metrics = {
		'acc': CategoricalAccuracy(),
		'ce': MetricWrapper(CrossEntropyWithVectors())
	}
	val_metrics_stack = {}
	return train_metrics, val_metrics, val_metrics_stack
