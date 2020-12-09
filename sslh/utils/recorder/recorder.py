import json
import math
import numpy as np

from sslh.utils.recorder.recorder_abc import RecorderABC

from time import time
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Union, Iterable


class Recorder(RecorderABC):
	def __init__(
		self,
		writer: Optional[SummaryWriter] = None,
		store_std: bool = False,
		store_mins: bool = False,
		store_maxs: bool = True,
		check_nan: bool = True,
	):
		self.writer = writer
		self.store_std = store_std
		self.store_mins = store_mins
		self.store_maxs = store_maxs
		self.check_nan = check_nan

		self._epoch_data = {}
		self._means = {}
		self._stds = {}
		self._idx_mins_means = {}
		self._idx_maxs_means = {}

		self.__start_time = time()
		self.__writer_to_update = False

	def start_record(self, epoch: int):
		pass

	def end_record(self, epoch: int):
		self._update_best_epoch(epoch)
		self._update_writer(epoch)
		self._epoch_data = {}

	def add_point(self, key: str, value: float):
		if self.check_nan and math.isnan(value):
			raise RuntimeError("Found NaN value for key \"{:s}\".".format(key))

		self.register_key(key)
		self._epoch_data[key].append(value)

	def set_point(self, key: str, value: float):
		self._epoch_data[key] = []
		self.add_point(key, value)

	def register_key(self, key: str):
		if key not in self._epoch_data.keys():
			self._epoch_data[key] = []

	def register_keys(self, keys: Iterable[str]):
		for key in keys:
			self.register_key(key)

	def is_key_registered(self, key: str) -> bool:
		return key in self._epoch_data.keys()

	def activate_auto_storage(self):
		self.store_mins = True
		self.store_maxs = True

	def deactivate_auto_storage(self):
		self.store_mins = False
		self.store_maxs = False

	def save_in_file(self, filepath: str, criterion: Union[str, Dict[str, str]] = "max"):
		content = {
			"duration": time() - self.__start_time,
			"all_bests": self.get_all_best_epoch(criterion),
		}

		with open(filepath, "w") as file:
			json.dump(content, file, indent="\t")

	def get_best_epoch(self, key: str, criterion: str = "max") -> Dict[str, Union[float, int]]:
		"""
			Returns a dictionary which contains the best mean and std for a specific metric.
		"""
		assert criterion in ["min", "max"]
		idx_best_means = self._idx_maxs_means if criterion == "max" else self._idx_mins_means
		if key not in idx_best_means.keys():
			raise RuntimeError("Unknown key \"{:s}\" in Recorder.".format(key))

		best_epoch = idx_best_means[key]
		best = {
			"best_epoch": best_epoch,
			"best_mean": self._means[key][best_epoch],
			"best_std": self._stds[key][best_epoch],
		}
		return best

	def get_all_best_epoch(self, criterion: Union[str, Dict[str, str]] = "max") -> Dict[str, Dict[str, Union[float, int]]]:
		if isinstance(criterion, str):
			idx_best_means = self._idx_maxs_means if criterion == "max" else self._idx_mins_means
		else:
			if set(self._means.keys()) != set(criterion.keys()):
				raise RuntimeError("Invalid criterion dict.")
			idx_best_means = {
				self._idx_maxs_means[key] if criterion[key] == "max" else self._idx_mins_means[key] for key in self._means.keys()
			}

		results = {}
		for key in idx_best_means.keys():
			best_epoch = idx_best_means[key]
			best_mean = self._means[key][best_epoch]
			best_std = self._stds[key][best_epoch]
			results.update({
				key: {
					"best_epoch": best_epoch,
					"best_mean": best_mean,
					"best_std": best_std,
				}
			})
		return results

	def get_writer(self) -> Optional[SummaryWriter]:
		return self.writer

	def get_epoch_data(self) -> Dict[str, List[float]]:
		return self._epoch_data

	def get_elapsed_time(self) -> float:
		return time() - self.__start_time

	def is_empty(self) -> bool:
		return len(self._means) == 0

	def _update_best_epoch(self, epoch: int):
		# Store final mean of the epoch
		for key, epoch_values in self._epoch_data.items():
			if key not in self._means.keys():
				self._means[key] = {}
			epoch_mean = np.mean(epoch_values)
			self._means[key][epoch] = epoch_mean

			if key not in self._stds.keys():
				self._stds[key] = {}
			epoch_std = np.std(epoch_values)
			self._stds[key][epoch] = epoch_std

			if key not in self._idx_mins_means.keys() or epoch_mean < self._means[key][self._idx_mins_means[key]]:
				self._idx_mins_means[key] = epoch

			if key not in self._idx_maxs_means.keys() or epoch_mean > self._means[key][self._idx_maxs_means[key]]:
				self._idx_maxs_means[key] = epoch

		self.__writer_to_update = True

	def get_current_means(self) -> Dict[str, float]:
		epoch_data = self.get_epoch_data()
		current_means = {key: np.mean(values) for key, values in epoch_data.items()}
		return current_means

	def _update_writer(self, epoch: int):
		if self.writer is not None and self.__writer_to_update:
			for key in self._epoch_data.keys():
				means = self._means[key]
				self.writer.add_scalar(key, means[epoch], epoch)

				stds = self._stds[key]
				if self.store_std:
					self.writer.add_scalar(key + "_std", stds[epoch], epoch)

				if "/" in key:
					section, metric_name = key.split("/")
				else:
					section, metric_name = "", key

				if self.store_mins:
					best_epoch = self._idx_mins_means[key]
					self.writer.add_scalar("{:s}_min/{:s}".format(section, metric_name), means[best_epoch], epoch)
					if self.store_std:
						self.writer.add_scalar("{:s}_min/{:s}_std".format(section, metric_name), stds[best_epoch], epoch)

				if self.store_maxs:
					best_epoch = self._idx_maxs_means[key]
					self.writer.add_scalar("{:s}_max/{:s}".format(section, metric_name), means[best_epoch], epoch)
					if self.store_std:
						self.writer.add_scalar("{:s}_max/{:s}_std".format(section, metric_name), stds[best_epoch], epoch)

			self.__writer_to_update = False
