
from sslh.utils.recorder.base import RecorderABC

from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict, List, Optional, Union


class NoRecorder(RecorderABC):
	def reset(self):
		pass

	def add_scalar(self, name: str, value: Union[Tensor, float]):
		pass

	def step(self):
		pass

	def set_storage(self, write_std: bool, write_min_mean: bool, write_max_mean: bool):
		pass

	def get_current_means(self) -> Dict[str, float]:
		return {}

	def get_min_max(self, name: str) -> (int, float, int, float):
		return 0, 0.0, 0, 0.0

	def get_current_names(self) -> List[str]:
		return []

	def get_all_names(self) -> List[str]:
		return []

	def get_writer(self) -> Optional[SummaryWriter]:
		return None
