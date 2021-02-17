
from argparse import Namespace

from mlu.datasets.utils import generate_split_samplers
from mlu.nn import OneHot

from sslh.datasets.base import DatasetBuilder
from sslh.datasets.dataset_sized import DatasetSized

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from typing import Any, Callable, Dict, Optional, List


class CIFAR10Builder(DatasetBuilder):
	def get_dataset_train(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = True,
		**kwargs,
	) -> DatasetSized:
		dataset = CIFAR10(
			dataset_root,
			train=True,
			download=download,
			transform=transform,
			target_transform=target_transform,
		)
		return dataset

	def get_dataset_val(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = True,
		**kwargs,
	) -> DatasetSized:
		dataset = CIFAR10(
			dataset_root,
			train=False,
			download=download,
			transform=transform,
			target_transform=target_transform,
		)
		return dataset

	def get_post_transform(self) -> Optional[Callable]:
		# Add postprocessing after each augmentation (shapes : [32, 32, 3] -> [3, 32, 32])
		return Compose([
			ToTensor(),
			Normalize(
				mean=(0.4914009, 0.48215896, 0.4465308),
				std=(0.24703279, 0.24348423, 0.26158753)
			),
		])

	def get_target_transform(self, smooth: Optional[float] = None) -> Optional[Callable]:
		return OneHot(self.get_nb_classes(), smooth)

	def get_dataset_name(self) -> str:
		return "CIFAR10"

	def get_data_type(self) -> str:
		return "image"

	def get_target_type(self) -> str:
		return "monolabel"

	def get_labels_names(self) -> Dict[int, str]:
		names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
		return {index: name for index, name in enumerate(names)}

	def get_folds(self) -> Optional[List[int]]:
		return None

	def has_evaluation(self) -> bool:
		return False
