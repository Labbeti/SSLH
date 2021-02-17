
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict, List, Optional, Union


class RecorderABC:
	def reset(self):
		"""
			Reset all values stored in Recorder.
		"""
		raise NotImplementedError("Abstract method")

	def add_scalar(self, name: str, value: Union[Tensor, float]):
		"""
			Add scalar value to recorder.

			:param name: The name of the scalar.
			:param value: The value (float or scalar tensor) to save.
		"""
		raise NotImplementedError("Abstract method")

	def step(self):
		"""
			Store data of the step. Should be called after all iterations of a training or a validation.
			Also store current step means in SummaryWriter and update min and max of all steps.
		"""
		raise NotImplementedError("Abstract method")

	def set_storage(self, write_mean: bool, write_std: bool, write_min_mean: bool, write_max_mean: bool):
		"""
			Activate or deactivate the automatic bests values in Recorder.

			:param write_mean: Write the mean of the current step.
			:param write_std: Write the std of the current step.
			:param write_min_mean: Write the min of all steps.
			:param write_max_mean: Write the max of all steps.
		"""
		raise NotImplementedError("Abstract method")

	def get_current_means(self) -> Dict[str, float]:
		"""
			:return: The continue mean of scalars stored in the current step.
		"""
		raise NotImplementedError("Abstract method")

	def get_min_max(self, name: str) -> (int, float, int, float):
		"""
			Returns the bests values for a metric.

			:param name: The name of the metric.
			:return: Returns a tuple (index of min, min value, index of max, max value).
		"""
		raise NotImplementedError("Abstract method")

	def get_current_names(self) -> List[str]:
		"""
			:return: Names of scalars stored in the recorder since the last step() call.
		"""
		raise NotImplementedError("Abstract method")

	def get_all_names(self) -> List[str]:
		"""
			:return: All names of scalars stored in the recorder.
		"""
		raise NotImplementedError("Abstract method")

	def get_writer(self) -> Optional[SummaryWriter]:
		"""
			:return: The optional tensorboard SummaryWriter used for saving data in disk.
		"""
		raise NotImplementedError("Abstract method")

	def add_scalars_dict(self, dict_scores: Dict[str, Union[Tensor, float]]):
		"""
			Add named scalars to Recorder.

			:param dict_scores: A dictionary of scalars.
		"""
		for name, value in dict_scores.items():
			self.add_scalar(name, value)

	def get_all_min_max(self) -> Dict[str, Dict[str, Union[int, float]]]:
		"""
			Get all best values for metrics stored.

			Ex: {"accuracy": {"idx_min": 0, "min": 2.0, "idx_max": 2, "max": 4.5}}

			:return: A dictionary containing the index of min, min value, index of max and max value for each global metric.
		"""
		all_min_max = {}
		for name in self.get_all_names():
			idx_min, min_, idx_max, max_ = self.get_min_max(name)
			all_min_max[name] = {
				"idx_min": idx_min,
				"min": min_,
				"idx_max": idx_max,
				"max": max_,
			}
		return all_min_max
