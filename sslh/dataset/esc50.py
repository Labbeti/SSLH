
from argparse import Namespace

from mlu.datasets.utils import generate_split_samplers
from mlu.datasets.wrappers import TransformDataset
from mlu.nn import Squeeze, OneHot

from sslh.dataset.abc import DatasetInterface
from sslh.dataset.module_call import ModuleCall
from sslh.dataset.dataset_sized import DatasetSized
from sslh.dataset.detail.esc_ds import ESC50_NoSR_Cached

from torch.nn import Sequential, Module
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from typing import Any, Callable, Dict, Optional, List


class ESC50Interface(DatasetInterface):
	def __init__(self):
		super().__init__()
		labels_list = [
			'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow', 'rain', 'sea_waves',
			'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 'wind', 'pouring_water', 'toilet_flush',
			'thunderstorm', 'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps', 'laughing',
			'brushing_teeth', 'snoring', 'drinking_sipping', 'door_wood_knock', 'mouse_click', 'keyboard_typing',
			'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick',
			'glass_breaking', 'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train', 'church_bells',
			'airplane', 'fireworks', 'hand_saw',
		]
		self._labels_names = {index: name for index, name in enumerate(labels_list)}
		self._sample_rate = 44100
		self._n_fft = 2048
		self._hop_length = 512
		self._n_mels = 64

	def get_dataset_train(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		folds: Optional[List[int]] = None,
		download: bool = True,
		**kwargs,
	) -> DatasetSized:
		if folds is None:
			folds = self.get_folds()
			folds = folds[:-1]

		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)

		dataset = ESC50_NoSR_Cached(root=dataset_root, folds=folds, download=download, transform=transform)
		dataset = TransformDataset(dataset, transform=target_transform, index=1)
		return dataset

	def get_dataset_val(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		folds: Optional[List[int]] = None,
		download: bool = True,
		**kwargs,
	) -> DatasetSized:
		if folds is None:
			folds = self.get_folds()
			folds = folds[-1:]

		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)

		dataset = ESC50_NoSR_Cached(root=dataset_root, folds=folds, download=download, transform=transform)
		dataset = TransformDataset(dataset, transform=target_transform, index=1)
		return dataset

	def get_split_samplers(
		self,
		dataset: DatasetSized,
		ratios: List[float],
		args: Optional[Namespace],
		**kwargs,
	) -> List[Dict[str, Any]]:
		samplers = generate_split_samplers(dataset, ratios, self.get_nb_classes())
		return [dict(sampler=sampler) for sampler in samplers]

	def get_base_transform(self) -> Optional[Callable]:
		return Sequential(
			MelSpectrogram(sample_rate=self._sample_rate, n_fft=self._n_fft, hop_length=self._hop_length, n_mels=self._n_mels),
			AmplitudeToDB(),
			Squeeze(),
		)

	def get_target_transform(self, smooth: Optional[float] = None) -> Optional[Callable]:
		return OneHot(self.get_nb_classes(), smooth)

	def get_dataset_name(self) -> str:
		return "ESC50"

	def get_data_type(self) -> str:
		return "audio"

	def get_target_type(self) -> str:
		return "monolabel"

	def get_labels_names(self) -> Dict[int, str]:
		return self._labels_names

	def get_folds(self) -> Optional[List[int]]:
		return list(range(1, 6))

	def has_evaluation(self) -> bool:
		return False
