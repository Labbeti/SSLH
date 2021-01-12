
import numpy as np
import random

from mlu.datasets.samplers import SubsetSampler
from mlu.datasets.utils import generate_indexes
from sslh.dataset.dataset_sized import DatasetSized
from torch.utils.data import Dataset, Subset
from typing import Any, Dict, List

DEFAULT_EPSILON = 2e-20


def generate_split_samplers(dataset: DatasetSized, ratios: List[float], nb_classes: int) -> List[Dict[str, Any]]:
	indexes = generate_indexes(dataset, nb_classes, ratios, target_one_hot=True)
	return [dict(sampler=SubsetSampler(dataset, indexes_split), batch_sampler=None) for indexes_split in indexes]
