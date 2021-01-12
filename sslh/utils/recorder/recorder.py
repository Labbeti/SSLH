
import math

from mlu.metrics import MinTracker, MaxTracker, IncrementalMean, IncrementalStd
from sslh.utils.recorder.base import RecorderABC

from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict, List, Optional, Union


class Recorder(RecorderABC):
	def __init__(
		self,
		writer: Optional[SummaryWriter] = None,
		write_std: bool = False,
		write_min_mean: bool = False,
		write_max_mean: bool = True,
		check_nan: bool = True,
	):
		super().__init__()
		self.writer = writer
		self.write_std = write_std
		self.write_min_mean = write_min_mean
		self.write_max_mean = write_max_mean
		self.check_nan = check_nan

		self._current_means = {}
		self._current_stds = {}

		self._mins_means = {}
		self._maxs_means = {}
		self._steps = {}

	def reset(self):
		self._current_means = {}
		self._current_stds = {}

		self._mins_means = {}
		self._maxs_means = {}
		self._steps = {}

	def add_scalar(self, name: str, value: Union[Tensor, float]):
		if self.check_nan and math.isnan(value):
			raise RuntimeError(f"Found NaN value for scalar {name}.")

		if name not in self._current_means.keys():
			self._current_means[name] = IncrementalMean()
			self._current_stds[name] = IncrementalStd()

		self._current_means[name].add(value)
		self._current_stds[name].add(value)

	def step(self):
		self._update_best()
		self._update_writer()

		self._current_means = {}
		self._current_stds = {}

	def set_storage(self, write_std: bool, write_min_mean: bool, write_max_mean: bool):
		self.write_std = write_std
		self.write_min_mean = write_min_mean
		self.write_max_mean = write_max_mean

	def set_writer(self, writer: Optional[SummaryWriter]):
		self.writer = writer

	def get_current_means(self) -> Dict[str, float]:
		return {name: mean.get_current().item() for name, mean in self._current_means.items()}

	def get_min_max(self, name: str) -> (int, float, int, float):
		min_tracker = self._mins_means[name]
		max_tracker = self._maxs_means[name]
		return (
			min_tracker.get_index(), min_tracker.get_current().item(),
			max_tracker.get_index(), max_tracker.get_current().item()
		)

	def get_current_names(self) -> List[str]:
		return list(self._current_means.keys())

	def get_all_names(self) -> List[str]:
		return list(self._mins_means.keys())

	def get_writer(self) -> Optional[SummaryWriter]:
		return self.writer

	def get_all_min_max(self) -> Dict[str, Dict[str, Union[int, float]]]:
		# TODO : rem ?
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

	def _update_best(self):
		new_names = set(self._current_means.keys()).difference(self._mins_means.keys())
		for name in new_names:
			self._mins_means[name] = MinTracker()
			self._maxs_means[name] = MaxTracker()

		for name, mean in self._current_means.items():
			self._mins_means[name].add(mean.get_current())
			self._maxs_means[name].add(mean.get_current())

	def _update_writer(self):
		if self.writer is not None:
			names = self.get_all_names()
			for name in names:
				if name in self._steps.keys():
					self._steps[name] += 1
				else:
					self._steps[name] = 0

			for name, mean in self._current_means.items():
				self.writer.add_scalar(name, mean.get_current(), self._steps[name])

			if self.write_std:
				suffix = "std"
				for name, std in self._current_stds.items():
					section, sub_name = name.split("/") if "/" in name else ("", name)
					self.writer.add_scalar(f"{section}_{suffix}/{sub_name}", std.get_current(), self._steps[name])

			if self.write_min_mean:
				suffix = "min"
				for name, mins in self._mins_means.items():
					section, sub_name = name.split("/") if "/" in name else ("", name)
					self.writer.add_scalar(f"{section}_{suffix}/{sub_name}", mins.get_current(), self._steps[name])

			if self.write_max_mean:
				suffix = "max"
				for name, maxs in self._maxs_means.items():
					section, sub_name = name.split("/") if "/" in name else ("", name)
					self.writer.add_scalar(f"{section}_{suffix}/{sub_name}", maxs.get_current(), self._steps[name])

			self.writer.flush()
