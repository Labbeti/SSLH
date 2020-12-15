
from ssl.utils.recorder.recorder_abc import RecorderABC

from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Union


class NoRecorder(RecorderABC):
	def start_record(self, epoch: int):
		pass

	def end_record(self, epoch: int):
		pass

	def add_point(self, name: str, value: float):
		pass

	def set_point(self, key: str, value: float):
		pass

	def register_key(self, key: str):
		pass

	def register_keys(self, keys: List[str]):
		pass

	def is_key_registered(self, key: str) -> bool:
		return False

	def save_in_file(self, filepath: str, criterion: Union[str, Dict[str, str]] = "max"):
		pass

	def get_best_epoch(self, key: str, criterion: str = "max") -> (int, float, float):
		return 0, 0.0, 0.0

	def get_all_best_epoch(self, criterion: Union[str, Dict[str, str]] = "max") -> Dict[str, Dict[str, float]]:
		return {}

	def get_writer(self) -> Optional[SummaryWriter]:
		return None

	def get_elapsed_time(self) -> float:
		return 0.0

	def get_current_means(self) -> Dict[str, float]:
		return {}
