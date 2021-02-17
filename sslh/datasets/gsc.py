
from mlu.datasets.wrappers import TransformDataset
from mlu.nn import OneHot
from mlu.transforms.waveform.pad import PadAlignLeft

from sslh.datasets.module_wrap import ModuleWrap
from sslh.datasets.base import DatasetBuilder
from sslh.datasets.dataset_sized import DatasetSized
from sslh.datasets.detail.gsc_ds import SpeechCommands, target_mapper

from torch.nn import Sequential, Module
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from typing import Callable, Dict, List, Optional


class GSCBuilder(DatasetBuilder):
	def __init__(self):
		super().__init__()
		self._labels_names = {idx: name for name, idx in target_mapper.items()}
		self._waveform_sec = 1
		self._sr = 16000
		self._n_fft = 2048  # window size
		self._hop_length = 512
		self._n_mels = 64

	def get_dataset_train(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = True,
		subset: str = "train",
		**kwargs,
	) -> DatasetSized:
		return get_gsc_dataset(dataset_root, transform, target_transform, download, subset)

	def get_dataset_val(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = True,
		subset: str = "validation",
		**kwargs,
	) -> DatasetSized:
		return get_gsc_dataset(dataset_root, transform, target_transform, download, subset)

	def get_dataset_eval(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = True,
		subset: str = "testing",
		**kwargs,
	) -> DatasetSized:
		return get_gsc_dataset(dataset_root, transform, target_transform, download, subset)

	def get_spec_transform(self) -> Optional[Callable]:
		return Sequential(
			PadAlignLeft(target_length=16000, fill_value=0.0),
			# Spec shape : (..., freq, time)
			MelSpectrogram(sample_rate=self._sr, n_fft=self._n_fft, hop_length=self._hop_length, n_mels=self._n_mels),
			AmplitudeToDB(),
		)

	def get_target_transform(self, smooth: Optional[float] = None) -> Optional[Callable]:
		return OneHot(self.get_nb_classes(), smooth)

	def get_dataset_name(self) -> str:
		return "GSC"

	def get_data_type(self) -> str:
		return "audio"

	def get_target_type(self) -> str:
		return "monolabel"

	def get_labels_names(self) -> Dict[int, str]:
		return self._labels_names

	def get_folds(self) -> Optional[List[int]]:
		return None

	def has_evaluation(self) -> bool:
		return True


def get_gsc_dataset(
	dataset_root: str,
	transform: Optional[Callable],
	target_transform: Optional[Callable],
	download: bool,
	subset: str,
) -> DatasetSized:
	if transform is not None and not isinstance(transform, Module):
		transform = ModuleWrap(transform)
	dataset = SpeechCommands(root=dataset_root, subset=subset, transform=transform, download=download)
	if target_transform is not None:
		dataset = TransformDataset(dataset, target_transform, index=1)
	return dataset
