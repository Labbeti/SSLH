
from argparse import Namespace

from sslh.augments.get_pool import get_pool_audio_with_name, add_transform_to_spec_for_pool
from sslh.augments.utils import Squeeze, PadUpTo
from sslh.datasets.module_call import ModuleCall
from sslh.datasets.abc import DatasetInterface
from sslh.datasets.dataset_sized import DatasetSized
from sslh.datasets.detail.gsc_ds import SpeechCommands, target_mapper
from sslh.datasets.transform import get_transform_self_supervised_flips
from sslh.datasets.wrappers.onehot_dataset import OneHotDataset
from sslh.models.wrn28_2 import WideResNet28Spec, WideResNet28RotSpec

from torch.nn import Sequential, Module
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchvision.transforms import RandomChoice

from typing import Callable, Optional, List


class GSCInterface(DatasetInterface):
	def __init__(self):
		super().__init__()
		self._class_names = {idx: name for name, idx in target_mapper.items()}

	def get_dataset_train_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)
		dataset_train = SpeechCommands(root=args.dataset_path, subset="train", transform=transform, download=True)
		dataset_train = OneHotDataset(dataset_train, self.get_nb_classes(), args.label_smoothing_value)
		return dataset_train

	def get_dataset_val_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)
		dataset_val = SpeechCommands(root=args.dataset_path, subset="validation", transform=transform, download=True)
		dataset_val = OneHotDataset(dataset_val, self.get_nb_classes())
		return dataset_val

	def get_dataset_eval_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> Optional[DatasetSized]:
		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)
		dataset_eval = SpeechCommands(root=args.dataset_path, subset="testing", transform=transform, download=True)
		dataset_eval = OneHotDataset(dataset_eval, self.get_nb_classes())
		return dataset_eval

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
		return "GSC"

	def get_nb_classes(self) -> int:
		return 35

	def get_nb_folds(self) -> Optional[int]:
		return None

	def get_default_model_name(self) -> str:
		return "WideResNet28Spec"

	def get_models(self) -> list:
		return [WideResNet28Spec, WideResNet28RotSpec]

	def get_transform_self_supervised(self, args: Optional[Namespace] = None) -> Callable:
		return get_transform_self_supervised_flips(args)

	def get_class_name(self, index: int) -> str:
		if index in self._class_names.keys():
			return self._class_names[index]
		else:
			raise RuntimeError(f"Invalid class index \"{index}\".")

	def _get_transform_base(self) -> Callable:
		transform_base = Sequential(
			PadUpTo(target_length=16000, mode="constant", value=0),
			MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=64),
			AmplitudeToDB(),
			Squeeze(),
		)
		return transform_base
