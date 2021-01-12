
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict, List, Optional, Union


class RecorderABC:
	def reset(self):
		raise NotImplementedError("Abstract method")

	def add_scalar(self, name: str, value: Union[Tensor, float]):
		raise NotImplementedError("Abstract method")

	def step(self):
		raise NotImplementedError("Abstract method")

	def set_storage(self, write_std: bool, write_min_mean: bool, write_max_mean: bool):
		raise NotImplementedError("Abstract method")

	def get_current_means(self) -> Dict[str, float]:
		raise NotImplementedError("Abstract method")

	def get_min_max(self, name: str) -> (int, float, int, float):
		"""
			Returns the bests values for a metric.

			:param name: The name of the metric.
			:return: Returns a tuple (index of min, min value, index of max, max value).
		"""
		raise NotImplementedError("Abstract method")

	def get_current_names(self) -> List[str]:
		raise NotImplementedError("Abstract method")

	def get_all_names(self) -> List[str]:
		raise NotImplementedError("Abstract method")

	def get_writer(self) -> Optional[SummaryWriter]:
		raise NotImplementedError("Abstract method")
