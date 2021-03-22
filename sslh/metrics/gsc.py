
from typing import Dict, Tuple
from mlu.metrics import Metric, CategoricalAccuracy, MetricWrapper
from mlu.nn import CrossEntropyWithVectors


def get_metrics_gsc() -> Tuple[Dict[str, Metric], Dict[str, Metric], Dict[str, Metric]]:
	train_metrics = {
		"acc": CategoricalAccuracy(),
	}
	val_metrics = {
		"acc": CategoricalAccuracy(),
		"ce": MetricWrapper(CrossEntropyWithVectors())
	}
	val_metrics_stack = {}
	return train_metrics, val_metrics, val_metrics_stack
