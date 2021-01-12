
from argparse import Namespace

from mlu.datasets.utils import generate_split_samplers
from mlu.datasets.wrappers import TransformDataset
from mlu.nn import Squeeze, OneHot

from sslh.augments.utils import PadUpTo
from sslh.dataset.module_call import ModuleCall
from sslh.dataset.abc import DatasetInterface
from sslh.dataset.dataset_sized import DatasetSized
from sslh.dataset.detail.gsc_ds import SpeechCommands, target_mapper

from torch.nn import Sequential, Module
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from typing import Any, Callable, Dict, List, Optional


class GSCInterface(DatasetInterface):
	def __init__(self):
		super().__init__()
		self._labels_names = {idx: name for name, idx in target_mapper.items()}

	def get_dataset_train(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = True,
		subset: str = "train",
		**kwargs,
	) -> DatasetSized:
		return _get_dataset(dataset_root, transform, target_transform, download, subset)

	def get_dataset_val(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = True,
		subset: str = "validation",
		**kwargs,
	) -> DatasetSized:
		return _get_dataset(dataset_root, transform, target_transform, download, subset)

	def get_dataset_eval(
		self,
		dataset_root: str,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = True,
		subset: str = "testing",
		**kwargs,
	) -> DatasetSized:
		return _get_dataset(dataset_root, transform, target_transform, download, subset)

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
			PadUpTo(target_length=16000, mode="constant", value=0),
			MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=64),
			AmplitudeToDB(),
			Squeeze(),
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


def _get_dataset(
	dataset_root: str,
	transform: Optional[Callable],
	target_transform: Optional[Callable],
	download: bool,
	subset: str,
) -> DatasetSized:
	if transform is not None and not isinstance(transform, Module):
		transform = ModuleCall(transform)
	dataset = SpeechCommands(root=dataset_root, subset=subset, transform=transform, download=download)
	dataset = TransformDataset(dataset, target_transform, index=1)
	return dataset
