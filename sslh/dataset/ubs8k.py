import os.path as osp

from argparse import Namespace

from mlu.datasets.utils import generate_split_samplers
from mlu.datasets.wrappers import TransformDataset
from mlu.nn import OneHot

from sslh.dataset.abc import DatasetInterface
from sslh.dataset.dataset_sized import DatasetSized
from sslh.dataset.transform import MelSpectrogramLibrosa, PowerToDbLibrosa

from torch.nn import Sequential
from torchvision.transforms import RandomChoice
from typing import Any, Callable, Dict, List, Optional

from ubs8k.datasets import Dataset as UBS8KDataset
from ubs8k.datasetManager import DatasetManager as UBS8KDatasetManager


class UBS8KManagerNoSpectrogram(UBS8KDatasetManager):
	def extract_feature(self, raw_data, **kwargs):
		return raw_data

	@property
	def validation(self):
		raise NotImplementedError


class UBS8KInterface(DatasetInterface):
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

		if transform is None:
			transform = ()
		elif isinstance(transform, RandomChoice):
			transform = tuple(transform.transforms)
		else:
			transform = (transform,)

		manager = self._load_manager(dataset_root)
		dataset = UBS8KDataset(manager, folds=folds, augments=transform, cached=False, augment_choser=lambda x: x)
		dataset = TransformDataset(dataset, target_transform, index=1)
		return dataset

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

		if transform is None:
			transform = ()
		elif isinstance(transform, RandomChoice):
			transform = tuple(transform.transforms)
		else:
			transform = (transform,)

		manager = self._load_manager(dataset_root)
		dataset = UBS8KDataset(manager, folds=folds, augments=transform, cached=False, augment_choser=lambda x: x)
		dataset = TransformDataset(dataset, target_transform, index=1)
		return dataset

	def get_split_samplers(
		self, dataset: DatasetSized, ratios: List[float], args: Optional[Namespace], **kwargs
	) -> List[Dict[str, Any]]:
		samplers = generate_split_samplers(dataset, ratios, self.get_nb_classes())
		return [dict(sampler=sampler) for sampler in samplers]

	def get_base_transform(self) -> Optional[Callable]:
		return Sequential(
			MelSpectrogramLibrosa(self._sr, self._n_fft, self._hop_length, self._n_mels),
			PowerToDbLibrosa(),
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

	# Private methods
	def _load_manager(self, dataset_path: str) -> UBS8KDatasetManager:
		if self._manager is None:
			metadata_root = osp.join(dataset_path, "metadata")
			audio_root = osp.join(dataset_path, "audio")
			self._manager = UBS8KManagerNoSpectrogram(metadata_root, audio_root)
		return self._manager
