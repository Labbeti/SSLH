
from pytorch_lightning.metrics import AveragePrecision
from typing import Dict, Tuple
from mlu.metrics import Metric, FScore, AveragePrecision, RocAuc, DPrime, BinaryAccuracy


def get_metrics_ads() -> Tuple[Dict[str, Metric], Dict[str, Metric], Dict[str, Metric]]:
	train_metrics = {
		"f1": FScore(threshold_input=0.5),
		"binacc": BinaryAccuracy(threshold_input=0.5),
	}
	val_metrics = {
		"f1": FScore(threshold_input=0.5),
		"binacc": BinaryAccuracy(threshold_input=0.5),
	}
	val_metrics_stack = {
		"f1": FScore(threshold_input=0.5),
		"binacc": BinaryAccuracy(threshold_input=0.5),
		"mAP": AveragePrecision(),
		"auc": RocAuc(),
		"dprime": DPrime(),
	}
	return train_metrics, val_metrics, val_metrics_stack
