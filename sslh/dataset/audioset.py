
import csv
import os.path as osp

from mlu.datasets.wrappers import TransformDataset
from mlu.transforms import ToTensor

from sslh.dataset.abc import DatasetInterface
from sslh.dataset.dataset_sized import DatasetSized
from sslh.dataset.detail.as_ds import (
	Audioset, ChunkAlignSampler, batch_balancer, BatchSamplerFromList, class_balance_split
)

from torch.nn import Sequential
from torch.utils.data.dataloader import DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from typing import Callable, Dict, Optional, List


class AudioSetInterface(DatasetInterface):
	def __init__(self):
		super().__init__()
		# Spectrogram of shape (64, 500)
		self._n_mels = 64
		self._n_time = 500
		self._sr = 32000
		self._n_fft = 2048
		self._hop_length = self._sr * 10 // self._n_time

		self._num_workers = 10
		self._pin_memory = False

		self._rdcc_nbytes = 512 * 1024 ** 2
		self._data_shape = (64, 500)
		self._data_key = "data"

		self._labels_names = {}
		self._read_meta_labels()

	def get_dataset_train(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		version: str = "unbalanced",
		**kwargs,
	) -> DatasetSized:
		assert version in ["balanced", "unbalanced"]
		dataset = Audioset(
			root=dataset_root,
			transform=transform,
			version=version,
			rdcc_nbytes=self._rdcc_nbytes,
			data_shape=self._data_shape,
			data_key=self._data_key,
		)
		if target_transform is not None:
			dataset = TransformDataset(dataset, transform=target_transform, index=1)
		return dataset

	def get_dataset_val(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		version: str = "eval",
		**kwargs,
	) -> DatasetSized:
		assert version in ["eval"]
		dataset = Audioset(
			root=dataset_root,
			transform=transform,
			version=version,
			rdcc_nbytes=self._rdcc_nbytes,
			data_shape=self._data_shape,
			data_key=self._data_key,
		)
		if target_transform is not None:
			dataset = TransformDataset(dataset, transform=target_transform, index=1)
		return dataset

	def get_base_transform(self) -> Optional[Callable]:
		return Sequential(
			ToTensor(),
			MelSpectrogram(sample_rate=self._sr, n_fft=self._n_fft, hop_length=self._hop_length, n_mels=self._n_mels),
			AmplitudeToDB(),
		)

	def get_target_transform(self, smooth: Optional[float] = None) -> Optional[Callable]:
		if smooth is not None and smooth > 0.0:
			raise NotImplementedError(f"Cannot use label smoothing with {self.get_dataset_name()} dataset.")
		return lambda x: x

	def get_dataset_name(self) -> str:
		return "AudioSet"

	def get_data_type(self) -> str:
		return "audio"

	def get_target_type(self) -> str:
		return "multilabel"

	def get_labels_names(self) -> Dict[int, str]:
		return self._labels_names

	def get_folds(self) -> Optional[List[int]]:
		return None

	def has_evaluation(self) -> bool:
		return False

	def get_loader_train(
		self,
		dataset: DatasetSized,
		batch_size: int,
		drop_last: bool = False,
		num_workers: int = 1,
		shuffle: bool = True,
		pin_memory: bool = False,
		**kwargs,
	) -> DataLoader:
		if not isinstance(dataset, Audioset):
			audioset = dataset
			while isinstance(audioset, TransformDataset):
				audioset = audioset.unwrap()

			if not isinstance(audioset, Audioset):
				raise ValueError("The dataset must be an AudioSet dataset.")
		else:
			audioset = dataset

		sampler_train = ChunkAlignSampler(audioset, batch_size=batch_size, shuffle=True)
		balancer = batch_balancer(batch_size=batch_size, pool_size=batch_size * 4)
		loader_train = DataLoader(
			dataset,
			batch_sampler=sampler_train,
			collate_fn=balancer,
			num_workers=num_workers,
			pin_memory=pin_memory,
		)

		return loader_train

	def get_loader_val(
		self,
		dataset: DatasetSized,
		batch_size: int,
		drop_last: bool = False,
		num_workers: int = 0,
		shuffle: bool = False,
		pin_memory: bool = False,
		**kwargs,
	) -> DataLoader:
		if not isinstance(dataset, Audioset):
			audioset = dataset
			while isinstance(audioset, TransformDataset):
				audioset = audioset.unwrap()

			if not isinstance(audioset, Audioset):
				raise ValueError("The dataset must be an AudioSet dataset.")
		else:
			audioset = dataset

		sampler_val = ChunkAlignSampler(audioset, batch_size=batch_size, shuffle=True)
		loader_val = DataLoader(
			dataset,
			batch_sampler=sampler_val,
			num_workers=num_workers,
			pin_memory=pin_memory,
		)
		return loader_val

	def get_loaders_split(
		self,
		labeled_dataset: DatasetSized,
		ratios: List[float],
		datasets: List[DatasetSized],
		batch_sizes: List[int],
		drop_last_list: List[bool],
		num_workers_list: List[int],
		target_transformed: bool = False,
		**kwargs,
	) -> List[DataLoader]:
		if not isinstance(labeled_dataset, Audioset):
			audioset = labeled_dataset
			while isinstance(audioset, TransformDataset):
				audioset = audioset.unwrap()

			if not isinstance(audioset, Audioset):
				raise ValueError("The dataset must be an AudioSet dataset.")
		else:
			audioset = labeled_dataset

		assert len(ratios) == len(datasets) == len(batch_sizes) == len(drop_last_list) == len(num_workers_list)
		assert 1 <= len(ratios) <= 2

		batch_size = sum(batch_sizes)
		balancer = batch_balancer(batch_size=batch_size, pool_size=batch_size * 4)

		unsupervised_ratio = ratios[1] if len(ratios) > 1 else None
		s_batches, u_batches = class_balance_split(
			audioset,
			supervised_ratio=ratios[0],
			unsupervised_ratio=unsupervised_ratio,
			batch_size=batch_size,
			verbose=False,
		)
		batch_sampler_s = BatchSamplerFromList(s_batches)
		batch_sampler_u = BatchSamplerFromList(u_batches)
		batch_samplers = [batch_sampler_s, batch_sampler_u]

		loaders = [
			DataLoader(
				dataset=dataset,
				num_workers=num_workers,
				batch_sampler=batch_sampler,
				collate_fn=balancer,
			)
			for dataset, num_workers, batch_sampler
			in zip(datasets, num_workers_list, batch_samplers)
		]
		return loaders

	# Private methods
	def _read_meta_labels(self):
		meta_filepath = osp.join(osp.dirname(__file__), "metadata", "as_labels.csv")
		if not osp.isfile(meta_filepath):
			raise RuntimeError(f"Cannot find meta file in path \"{meta_filepath}.")

		with open(meta_filepath, "r") as table_file:
			reader = csv.reader(table_file, skipinitialspace=True, strict=True)

			for _ in range(2):
				next(reader)

			self._labels_names = {}
			for info in reader:
				index, mid, name = info
				index = int(index)
				self._labels_names[index] = name
