
from typing import Iterable, Protocol


class Callback(Protocol):
	"""
		Protocol base class for callbacks. A callback is a object that implements a "step()" method.
	"""
	def step(self):
		raise NotImplementedError("Protocol method")


class Caller:
	def __init__(self):
		"""
			Base class for adding callbacks in Trainers and Validators.

			Callbacks must implements a "step()" method.
		"""
		self.callbacks_on_start = []
		self.callbacks_on_end = []

	def _on_start(self):
		"""
			Call step() method on callbacks added for start.
		"""
		for callback in self.callbacks_on_start:
			callback.step()

	def _on_end(self):
		"""
			Call step() method on callbacks added for end.
		"""
		for callback in self.callbacks_on_end:
			callback.step()

	def add_callback_on_start(self, callback: Callback):
		"""
			Add a callback that will be called before the main process (training or validation).

			:param callback: The callback object to add.
		"""
		if not hasattr(callback, "step"):
			raise RuntimeError("The callback must implements a step() method.")
		self.callbacks_on_start.append(callback)

	def add_callback_on_end(self, callback: Callback):
		"""
			Add a callback that will be called after the main process (training or validation).

			:param callback: The callback object to add.
		"""
		if not hasattr(callback, "step"):
			raise RuntimeError("The callback must implements a step() method.")
		self.callbacks_on_end.append(callback)

	def add_callback_list_on_start(self, callbacks: Iterable[Callback]):
		"""
			Add a list of callbacks that will be called before the main process (training or validation).

			:param callbacks: The list of callback objects to add.
		"""
		for callback in callbacks:
			self.add_callback_on_start(callback)

	def add_callback_list_on_end(self, callbacks: Iterable[Callback]):
		"""
			Add a list of callbacks that will be called after the main process (training or validation).

			:param callbacks: The list of callback objects to add.
		"""
		for callback in callbacks:
			self.add_callback_on_end(callback)
