
from abc import ABC
from typing import Dict, List


class DisplayABC(ABC):
	"""
		Abstract class for displays.
		A display is a simple class that print current metrics and losses values in a specific format.
	"""

	def print_header(self, name: str, keys: List[str], add_end_line_at_start: bool = True):
		"""
			Print the names (keys) of the current training in a specific format.

			:param name: The name of the training or validation (train, val or eval).
			:param keys: The columns names to print at the beginning of the loop.
			:param add_end_line_at_start: Print a newline just before the keys.
		"""
		raise NotImplementedError("Abstract method")

	def print_current_values(self, current_means: Dict[str, float], iteration: int, nb_iterations: int, epoch: int):
		"""
			Print current values of the iteration.

			:param current_means: Continue average of all metrics.
			:param iteration: Current iteration number.
			:param nb_iterations: Number of iterations for the training/validation loop.
			:param epoch: Current epoch of the program.
		"""
		raise NotImplementedError("Abstract method")
