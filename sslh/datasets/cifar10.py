
from argparse import Namespace

from mlu.utils.convert import ToNumpy

from sslh.augments.get_pool import get_pool_img_with_name, add_process_for_pool
from sslh.datasets.abc import DatasetInterface
from sslh.datasets.dataset_sized import DatasetSized
from sslh.datasets.transform import get_transform_self_supervised_rotate
from mlu.datasets.wrappers import OneHotDataset
from sslh.models.wrn28_2 import WideResNet28, WideResNet28Rot
from sslh.models.vgg import VGGRot, VGG11Rot

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomChoice, Normalize, ToTensor
from typing import Callable, Optional, List


class CIFAR10Interface(DatasetInterface):
	def get_dataset_train_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		dataset_train = CIFAR10(args.dataset_path, train=True, download=True, transform=transform)
		dataset_train = OneHotDataset(dataset_train, self.get_nb_classes(), args.label_smoothing_value)
		return dataset_train

	def get_dataset_val_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		dataset_val = CIFAR10(args.dataset_path, train=False, download=True, transform=transform)
		dataset_val = OneHotDataset(dataset_val, self.get_nb_classes())
		return dataset_val

	def get_dataset_eval_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> Optional[DatasetSized]:
		return None

	def get_transform_augm_none(self, args: Optional[Namespace]) -> Callable:
		pre_process_fn, post_process_fn = self._get_pre_post_processes()
		pool_none = get_pool_img_with_name(args.augm_none, args)
		pool_none = add_process_for_pool(pool_none, pre_process=pre_process_fn, post_process=post_process_fn)
		return RandomChoice(pool_none)

	def get_transform_augm_weak(self, args: Optional[Namespace]) -> Callable:
		pre_process_fn, post_process_fn = self._get_pre_post_processes()
		pool_weak = get_pool_img_with_name(args.augm_weak, args)
		pool_weak = add_process_for_pool(pool_weak, pre_process=pre_process_fn, post_process=post_process_fn)
		return RandomChoice(pool_weak)

	def get_transform_augm_strong(self, args: Optional[Namespace]) -> Callable:
		pre_process_fn, post_process_fn = self._get_pre_post_processes()
		pool_strong = get_pool_img_with_name(args.augm_strong, args)
		pool_strong = add_process_for_pool(pool_strong, pre_process=pre_process_fn, post_process=post_process_fn)
		return RandomChoice(pool_strong)

	def get_transform_val(self, args: Optional[Namespace]) -> Callable:
		pre_process_fn, post_process_fn = self._get_pre_post_processes()
		pool_val = Compose([pre_process_fn, post_process_fn])
		return pool_val

	def get_dataset_name(self) -> str:
		return "CIFAR10"

	def get_nb_classes(self) -> int:
		return 10

	def get_nb_folds(self) -> Optional[int]:
		return None

	def get_default_model_name(self) -> str:
		return "WideResNet28"

	def get_models(self) -> list:
		return [WideResNet28, WideResNet28Rot, VGGRot, VGG11Rot]

	def get_transform_self_supervised(self, args: Optional[Namespace] = None) -> Callable:
		return get_transform_self_supervised_rotate(args)

	def get_class_name(self, index: int) -> str:
		# TODO : check classes order
		return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"][index]

	def _get_pre_post_processes(self) -> (Optional[Callable], Optional[Callable]):
		# Add preprocessing before each augmentation
		pre_process_fn = None

		# Add postprocessing after each augmentation (shape : [32, 32, 3] -> [3, 32, 32])
		post_process_fn = Compose([
			ToTensor(),
		])
		return pre_process_fn, post_process_fn
