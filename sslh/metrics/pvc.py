
from typing import Dict, Tuple
from mlu.metrics import Metric, CategoricalAccuracy, UAR, AveragePrecision


def get_metrics_pvc() -> Tuple[Dict[str, Metric], Dict[str, Metric], Dict[str, Metric]]:
	train_metrics = {
		"acc": CategoricalAccuracy(),
	}
	val_metrics = {
		"acc": CategoricalAccuracy(),
	}
	val_metrics_stack = {
		"acc": CategoricalAccuracy(),
		"mAP": AveragePrecision(),
		"uar": UAR(),
	}
	return train_metrics, val_metrics, val_metrics_stack
