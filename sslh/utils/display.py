
from sslh.utils.display_abc import DisplayABC
from time import time
from typing import Dict, List, Iterable


class ColumnDisplay(DisplayABC):
	KEY_MAX_LENGTH = 10

	def __init__(self):
		self._epoch_start_date = time()
		self._previous_name = ""
		self._previous_keys = {}

	def print_header(self, name: str, keys: List[str], add_end_line_at_start: bool = True):
		def filter_name(key_name: str) -> str:
			if len(key_name) <= ColumnDisplay.KEY_MAX_LENGTH:
				return key_name.center(ColumnDisplay.KEY_MAX_LENGTH)
			else:
				return key_name[:ColumnDisplay.KEY_MAX_LENGTH]

		content = ["{:s}".format(name.center(16))]
		content += [filter_name(metric_name) for metric_name in sorted(keys)]
		content += ["took (s)".center(ColumnDisplay.KEY_MAX_LENGTH)]

		if add_end_line_at_start:
			print("")
		print("- {:s} -".format(" - ".join(content)))

		self._epoch_start_date = time()
		self._previous_name = name
		self._previous_keys = {"{:s}/{:s}".format(name, key) for key in keys}

	def print_current_values(self, current_means: Dict[str, float], iteration: int, nb_iterations: int, epoch: int):
		self._check_with_previous_keys(current_means.keys())
		progression = int(100 * (iteration + 1) / nb_iterations)

		content = ["Epoch {:3d} - {:3d}%".format(epoch + 1, progression)]
		content += [
			("{:.4e}".format(mean).center(ColumnDisplay.KEY_MAX_LENGTH))
			for _metric_name, mean in sorted(current_means.items())
		]
		content += ["{:.2f}".format(time() - self._epoch_start_date).center(ColumnDisplay.KEY_MAX_LENGTH)]

		print("- {:s} -".format(" - ".join(content)), end="\r")

		if iteration == nb_iterations - 1:
			print("")

	def _check_with_previous_keys(self, keys: Iterable[str]):
		if set(self._previous_keys) != set(keys):
			raise RuntimeError("Missing a metric name when print current values : {:s} != {:s}.".format(
				str(set(self._previous_keys)), str(set(keys))))


class NoDisplay(DisplayABC):
	def print_header(self, name: str, keys: List[str], add_end_line_at_start: bool = True):
		pass

	def print_current_values(self, current_means: Dict[str, float], iteration: int, nb_iterations: int, epoch: int):
		pass


class LineDisplay(DisplayABC):
	def __init__(self):
		self._prev_name = ""

	def print_header(self, name: str, keys: List[str], add_end_line_at_start: bool = True):
		self._prev_name = name
		if add_end_line_at_start:
			print("")

	def print_current_values(self, current_means: Dict[str, float], iteration: int, nb_iterations: int, epoch: int):
		progression = int(100 * (iteration + 1) / nb_iterations)
		content = ", ".join(["{:s}: {:.4e}".format(name, mean) for name, mean in current_means.items()])
		print("{:5s}, epoch {:3d}, {:3d}%, {:s}".format(self._prev_name, epoch, progression, content), end="\r")

		if iteration == nb_iterations - 1:
			print("")
