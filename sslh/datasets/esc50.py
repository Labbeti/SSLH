
from argparse import Namespace

from sslh.augments.get_pool import get_pool_audio_with_name, add_transform_to_spec_for_pool
from sslh.augments.utils import Squeeze

from sslh.datasets.abc import DatasetInterface
from sslh.datasets.module_call import ModuleCall
from sslh.datasets.dataset_sized import DatasetSized
from sslh.datasets.detail.esc_ds import ESC50_NoSR_Cached
from sslh.datasets.transform import get_transform_self_supervised_flips
from sslh.datasets.wrappers.onehot_dataset import OneHotDataset
from sslh.models.wrn28_2 import WideResNet28RotSpec, WideResNet28Spec

from torch.nn import Sequential, Module
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import RandomChoice, Compose
from typing import Callable, Optional, List


class ESC50Interface(DatasetInterface):
	def __init__(self):
		super().__init__()
		self._class_names = [
			'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow', 'rain', 'sea_waves',
			'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 'wind', 'pouring_water', 'toilet_flush',
			'thunderstorm', 'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps', 'laughing',
			'brushing_teeth', 'snoring', 'drinking_sipping', 'door_wood_knock', 'mouse_click', 'keyboard_typing',
			'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick',
			'glass_breaking', 'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train', 'church_bells',
			'airplane', 'fireworks', 'hand_saw',
		]
		self._sample_rate = 44100

	def get_dataset_train_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		if folds is None:
			folds = (1, 2, 3, 4)
		transform = ModuleCall(transform)
		dataset_train = ESC50_NoSR_Cached(root=args.dataset_path, folds=folds, download=True, transform=transform)
		dataset_train = OneHotDataset(dataset_train, self.get_nb_classes(), smooth=args.label_smoothing_value)
		return dataset_train

	def get_dataset_val_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		if folds is None:
			folds = (5,)
		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)
		dataset_val = ESC50_NoSR_Cached(root=args.dataset_path, folds=folds, download=True, transform=transform)
		dataset_val = OneHotDataset(dataset_val, self.get_nb_classes(), smooth=None)
		return dataset_val

	def get_dataset_eval_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> Optional[DatasetSized]:
		return None

	def get_transform_augm_none(self, args: Optional[Namespace]) -> Callable:
		return self._get_transform(args.augm_none, args)

	def get_transform_augm_weak(self, args: Optional[Namespace]) -> Callable:
		return self._get_transform(args.augm_weak, args)

	def get_transform_augm_strong(self, args: Optional[Namespace]) -> Callable:
		return self._get_transform(args.augm_strong, args)

	def get_transform_val(self, args: Optional[Namespace]) -> Callable:
		return self._get_transform_base()

	def get_dataset_name(self) -> str:
		return "ESC50"

	def get_nb_folds(self) -> Optional[int]:
		return 5

	def get_nb_classes(self) -> int:
		return 50

	def get_default_model_name(self) -> str:
		return "WideResNet28Spec"

	def get_models(self) -> list:
		return [WideResNet28Spec, WideResNet28RotSpec]

	def get_transform_self_supervised(self, args: Optional[Namespace] = None) -> Callable:
		return get_transform_self_supervised_flips(args)

	def get_class_name(self, index: int) -> str:
		if 0 <= index < len(self._class_names):
			return self._class_names[index]
		else:
			raise RuntimeError(f"Invalid class index \"{index}\".")

	def _get_transform_base(self) -> Callable:
		transform_base = Compose([
			Sequential(
				MelSpectrogram(sample_rate=self._sample_rate, n_fft=2048, hop_length=512, n_mels=64),
				AmplitudeToDB(),
			),
			Squeeze(),
		])
		return transform_base

	def _get_transform(self, augm_name: str, args: Optional[Namespace]) -> Callable:
		transform_base = self._get_transform_base()
		augm_pool = get_pool_audio_with_name(augm_name, args)
		augm_pool = add_transform_to_spec_for_pool(augm_pool, transform_base)
		return RandomChoice(augm_pool)
