
from argparse import Namespace

from sslh.augments.get_pool import get_pool_audio_with_name, add_transform_to_spec_for_pool
from sslh.datasets.module_call import ModuleCall
from sslh.datasets.abc import DatasetInterface
from sslh.datasets.dataset_sized import DatasetSized
from sslh.datasets.transform import get_transform_self_supervised_flips
from sslh.models.wrn28_2 import WideResNet28RotSpec, WideResNet28Spec
from tas.dataset import AudioSetDataset, Subset

from torch.nn import Sequential, Module
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import RandomChoice
from typing import Optional, Callable, List


class AudioSetInterface(DatasetInterface):
	def get_dataset_train_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)
		dataset = AudioSetDataset(args.dataset_path, Subset.BALANCED, transform=transform)
		return dataset

	def get_dataset_val_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)
		dataset = AudioSetDataset(args.dataset_path, Subset.EVAL, transform=transform)
		return dataset

	def get_dataset_eval_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> Optional[DatasetSized]:
		return None

	def get_transform_augm_none(self, args: Optional[Namespace]) -> Callable:
		transform_base = self._get_transform_base()
		pool_weak = get_pool_audio_with_name(args.augm_none, args)
		pool_weak = add_transform_to_spec_for_pool(pool_weak, transform_base)
		return RandomChoice(pool_weak)

	def get_transform_augm_weak(self, args: Optional[Namespace]) -> Callable:
		transform_base = self._get_transform_base()
		pool_weak = get_pool_audio_with_name(args.augm_weak, args)
		pool_weak = add_transform_to_spec_for_pool(pool_weak, transform_base)
		return RandomChoice(pool_weak)

	def get_transform_augm_strong(self, args: Optional[Namespace]) -> Callable:
		transform_base = self._get_transform_base()
		pool_strong = get_pool_audio_with_name(args.augm_strong, args)
		pool_strong = add_transform_to_spec_for_pool(pool_strong, transform_base)
		return RandomChoice(pool_strong)

	def get_transform_val(self, args: Optional[Namespace]) -> Callable:
		return self._get_transform_base()

	def get_dataset_name(self) -> str:
		return "AudioSet"

	def get_nb_classes(self) -> int:
		return 527

	def get_nb_folds(self) -> Optional[int]:
		return None

	def get_default_model_name(self) -> str:
		return "WideResNet28Spec"

	def get_models(self) -> list:
		return [WideResNet28Spec, WideResNet28RotSpec]

	def get_transform_self_supervised(self, args: Optional[Namespace] = None) -> Callable:
		return get_transform_self_supervised_flips(args)

	def get_class_name(self, index: int) -> str:
		raise NotImplementedError("TODO")

	def _get_transform_base(self) -> Callable:
		# TODO : add argument for mel spec
		transform_base = Sequential(
			MelSpectrogram(),
			AmplitudeToDB(),
		)
		return transform_base
