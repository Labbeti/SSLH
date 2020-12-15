
from abc import ABC
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Union


class RecorderABC(ABC):
	"""
		Wrapper of SummaryWriter for saving data in tensorboard and compute min and max validation metrics.
	"""

	def start_record(self, epoch: int):
		raise NotImplementedError("Abstract method")

	def end_record(self, epoch: int):
		raise NotImplementedError("Abstract method")

	def add_point(self, name: str, value: float):
		raise NotImplementedError("Abstract method")

	def set_point(self, key: str, value: float):
		raise NotImplementedError("Abstract method")

	def register_key(self, key: str):
		raise NotImplementedError("Abstract method")

	def register_keys(self, keys: List[str]):
		raise NotImplementedError("Abstract method")

	def is_key_registered(self, key: str) -> bool:
		raise NotImplementedError("Abstract method")

	def save_in_file(self, filepath: str, criterion: Union[str, Dict[str, str]] = "max"):
		raise NotImplementedError("Abstract method")

	def get_best_epoch(self, key: str, criterion: str = "max") -> (int, float, float):
		raise NotImplementedError("Abstract method")

	def get_all_best_epoch(self, criterion: Union[str, Dict[str, str]] = "max") -> Dict[str, Dict[str, float]]:
		raise NotImplementedError("Abstract method")

	def get_writer(self) -> Optional[SummaryWriter]:
		raise NotImplementedError("Abstract method")

	def get_elapsed_time(self) -> float:
		raise NotImplementedError("Abstract method")

	def get_current_means(self) -> Dict[str, float]:
		raise NotImplementedError("Abstract method")
