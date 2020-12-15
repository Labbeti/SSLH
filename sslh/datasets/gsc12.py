from argparse import Namespace
from typing import Optional, Callable, List

from sslh.datasets.dataset_sized import DatasetSized
from sslh.datasets.gsc import GSCInterface
from sslh.datasets.detail.gsc_ds import SpeechCommand10
from sslh.datasets.module_call import ModuleCall
from mlu.datasets.wrappers import OneHotDataset
from torch.nn import Module


class GSC12Interface(GSCInterface):
	def __init__(self):
		super().__init__()
		self._class_names = [
			"yes", "no", "up", "down", "left", "right", "off", "on", "go", "stop", "silence", "_background_noise_"
		]

	def get_dataset_train_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)

		dataset_train = SpeechCommand10(
			root=args.dataset_path, subset="train", transform=transform, download=True, percent_to_drop=0.5)
		dataset_train = OneHotDataset(dataset_train, self.get_nb_classes(), args.label_smoothing_value)
		return dataset_train

	def get_dataset_val_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> DatasetSized:
		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)

		dataset_val = SpeechCommand10(
			root=args.dataset_path, subset="validation", transform=transform, download=True, percent_to_drop=0.0)
		dataset_val = OneHotDataset(dataset_val, self.get_nb_classes())
		return dataset_val

	def get_dataset_eval_with_transform(self, args: Namespace, folds: Optional[List[int]], transform: Optional[Callable]) -> Optional[DatasetSized]:
		if transform is not None and not isinstance(transform, Module):
			transform = ModuleCall(transform)

		dataset_eval = SpeechCommand10(
			root=args.dataset_path, subset="testing", transform=transform, download=True, percent_to_drop=0.0)
		dataset_eval = OneHotDataset(dataset_eval, self.get_nb_classes())
		return dataset_eval

	def get_dataset_name(self) -> str:
		return "GSC12"

	def get_nb_classes(self) -> int:
		# 10 classes + 1 class unknown + 1 class silence
		return 12

	def get_class_name(self, index: int) -> str:
		if 0 <= index < len(self._class_names):
			return self._class_names[index]
		else:
			raise RuntimeError(f"Invalid class index \"{index}\".")
