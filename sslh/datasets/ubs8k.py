
import numpy as np
import os.path as osp
import torch

from mlu.datasets.wrappers import TransformDataset
from mlu.nn import OneHot, UnSqueeze
from mlu.transforms import ToTensor, ToNumpy, Compose
from mlu.transforms.waveform import PadCrop

from sslh.augments.utils import MelSpectrogramLibrosa, PowerToDbLibrosa
from sslh.datasets.base import DatasetBuilder
from sslh.datasets.dataset_sized import DatasetSized

from typing import Callable, Dict, List, Optional

from ubs8k.datasets import Dataset as UBS8KDataset
from ubs8k.datasetManager import DatasetManager as UBS8KDatasetManager


class UBS8KDatasetNoSpecNoPad(UBS8KDataset):
	"""
		UBS8K dataset without pad, crop and cache.
	"""

	def _pad_and_crop(self, raw_data):
		return raw_data

	def _generate_data(self, index: int):
		# load the raw_audio
		filename = self.filenames[index]
		raw_audio = self.x[filename]

		# recover ground truth
		y = self.y.at[filename, "classID"]

		# check if augmentation should be applied
		apply_augmentation = self.augment_S if index in self.s_idx else self.augment_U

		# chose augmentation, if no return an empty list
		augment_fn = self.augment_choser(self.augments) if self.augments else []

		# Apply augmentation, only one that applies on the signal will be executed
		audio_transformed, cache_id = self._apply_augmentation(raw_audio, augment_fn, filename, apply_augmentation)
		y = np.asarray(y)

		# call end of generation callbacks
		self.end_of_generation_callback()

		return audio_transformed, y


class UBS8KBuilder(DatasetBuilder):
	def __init__(self):
		super().__init__()
		labels_names_list = [
			"air_conditioner",
			"car_horn",
			"children_playing",
			"dog_bark",
			"drilling",
			"engine_idling",
			"gun_shot",
			"jackhammer",
			"siren",
			"street_music",
		]
		self._labels_names = {index: name for index, name in enumerate(labels_names_list)}
		self._manager = None

		self._sr = 22050
		self._n_fft = 2048
		self._hop_length = 512
		self._n_mels = 64

	def get_dataset_train(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		folds: Optional[List[int]] = None,
		**kwargs,
	) -> DatasetSized:
		if folds is None:
			folds = self.get_folds()
			folds = folds[:-1]

		self._load_manager(dataset_root)
		return get_ubs8k_dataset(
			self._manager,
			folds,
			transform,
			target_transform,
		)

	def get_dataset_val(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		folds: Optional[List[int]] = None,
		**kwargs,
	) -> DatasetSized:
		if folds is None:
			folds = self.get_folds()
			folds = folds[-1:]

		self._load_manager(dataset_root)
		return get_ubs8k_dataset(
			self._manager,
			folds,
			transform,
			target_transform,
		)

	def get_pre_transform(self) -> Optional[Callable]:
		return Compose(
			ToTensor(dtype=torch.float),
			PadCrop(target_length=self._sr * UBS8KDatasetManager.LENGTH, align="left"),
		)

	def get_post_transform(self) -> Optional[Callable]:
		return UnSqueeze(dim=0)

	def get_spec_transform(self) -> Optional[Callable]:
		return Compose(
			ToNumpy(),
			MelSpectrogramLibrosa(self._sr, self._n_fft, self._hop_length, self._n_mels),
			PowerToDbLibrosa(),
			ToTensor(dtype=torch.float),
		)

	def get_target_transform(self, smooth: Optional[float] = None) -> Optional[Callable]:
		return OneHot(self.get_nb_classes(), smooth)

	def get_dataset_name(self) -> str:
		return "UBS8K"

	def get_data_type(self) -> str:
		return "audio"

	def get_target_type(self) -> str:
		return "monolabel"

	def get_labels_names(self) -> Dict[int, str]:
		return self._labels_names

	def get_folds(self) -> Optional[List[int]]:
		return list(range(1, 11))

	def has_evaluation(self) -> bool:
		return False

	# Other methods
	def get_manager(self) -> UBS8KDatasetManager:
		return self._manager

	def _load_manager(self, dataset_path: str):
		if self._manager is None:
			if not osp.isdir(dataset_path):
				raise RuntimeError(f"Unknown directory '{dataset_path}'.")
			metadata_root = osp.join(dataset_path, "metadata")
			audio_root = osp.join(dataset_path, "audio")
			self._manager = UBS8KDatasetManager(metadata_root, audio_root)


def get_ubs8k_dataset(
	manager: UBS8KDatasetManager,
	folds: List[int],
	transform: Optional[Callable] = None,
	target_transform: Optional[Callable] = None,
) -> DatasetSized:
	if transform is None:
		transform = ()
	else:
		transform = (transform,)

	if not isinstance(folds, tuple):
		folds = tuple(folds)

	dataset = UBS8KDatasetNoSpecNoPad(
		manager, folds=folds, augments=transform, cached=False, augment_choser=lambda x: x)
	if target_transform is not None:
		dataset = TransformDataset(dataset, target_transform, index=1)
	return dataset
