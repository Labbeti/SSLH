
from typing import Optional


class WarmUp:
	"""
		warmup class.

		Linearly increase a value each time the method "step()" is called.
		Access the current value with method "value()".
		If nb_steps == 0, the coefficient will be 1.0 and the value will be always the max value expected.
	"""
	def __init__(
		self,
		max_value: float,
		nb_steps: int,
		obj: Optional[object] = None,
		attr_name: Optional[str] = None,
		min_value: float = 0.0,
	):
		self._max_value = max_value
		self._nb_steps = nb_steps
		self._obj = obj
		self._attr_name = attr_name
		self._min_value = min_value

		self._cur_step = 0

		self._check_attributes()
		self._update_obj()

	def reset(self):
		self._cur_step = 0
		self._update_obj()

	def set_obj(self, obj: Optional[object]):
		self._obj = obj
		self._check_attributes()
		self._update_obj()

	def step(self):
		if self._cur_step < self._nb_steps:
			self._cur_step += 1
			self._update_obj()

	def value(self) -> float:
		return (self._max_value - self._min_value) * self.get_coefficient() + self._min_value

	def get_value(self) -> float:
		return self.value()

	def get_coefficient(self) -> float:
		if self._nb_steps > 0:
			return self._cur_step / self._nb_steps
		else:
			return 1.0

	def _update_obj(self):
		if self._obj is not None:
			self._obj.__setattr__(self._attr_name, self.value())

	def _check_attributes(self):
		if self._obj is not None and not hasattr(self._obj, self._attr_name):
			raise RuntimeError(
				"Use warmup on attribute \"{:s}\" but the object \"{:s}\" do not contains this attribute.".format(
					self._attr_name, self._obj.__class__.__name__))
