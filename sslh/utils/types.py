
import math

from typing import Iterable, Optional, Sized, Union, Protocol, Callable


class IterableSized(Iterable, Sized, Protocol):
	"""
		Abstract class for an Iterable and Sized type.
	"""
	pass


def str_to_bool(x: str) -> bool:
	"""
		Convert a string to bool. Case insensitive.
		:param x:
			x in ["true", "1", "yes", "y"] => True
			x in ["false", "0", "no", "n"] => False
			_ => RuntimeError
		:returns: The corresponding boolean value.
	"""
	x_low = str(x).lower()
	if x_low in ["true", "1", "yes", "y"]:
		return True
	elif x_low in ["false", "0", "no", "n"]:
		return False
	else:
		raise RuntimeError("Invalid boolean argument \"{:s}\".".format(x))


def str_to_optional_bool(x: str) -> Optional[bool]:
	"""
		Convert a string to optional bool value. Case insensitive.
		:param x:
			x in ["none"] => None
			x in ["true", "1", "yes", "y"] => True
			x in ["false", "0", "no", "n"] => False
			_ => RuntimeError
		:returns: The corresponding boolean or None value.
	"""
	x_low = str(x).lower()
	if x_low in ["none"]:
		return None
	else:
		return str_to_bool(x)


def str_to_optional_str(x: str) -> Optional[str]:
	"""
		Convert string to optional string value. Case insensitive.
		:param x: Any string value.
		:returns: None if x == "None", otherwise the string value.
	"""
	x_low = str(x).lower()
	if x_low in ["none"]:
		return None
	else:
		return x


def str_to_optional_int(x: str) -> Optional[int]:
	"""
		Convert string to optional integer value. Case insensitive.
		:param x: Any string value.
		:returns: Integer value, None or throw ValueError exception.
	"""
	x_low = str(x).lower()
	if x_low in ["none"]:
		return None
	else:
		return int(x)


def str_to_optional_float(x: str) -> Optional[float]:
	"""
		Convert string to optional float value. Case insensitive.
		:param x: Any string value.
		:returns: Float value, None or throw ValueError exception.
	"""
	x = str(x)
	if x.lower() == "none":
		return None
	else:
		return float(x)


def str_to_union_str_int(x: str) -> Union[str, int]:
	"""
		Convert string to integer value or string value.
		:param x: Any string value.
		:returns: If x is digit, return the integer value, otherwise returns a the same string value.
	"""
	x = str(x)
	try:
		x_int = int(x)
		return x_int
	except ValueError:
		return x


def float_in_range(
	min_: float,
	max_: float,
	include_min: bool = True,
	include_max: bool = False,
) -> Callable[[str], float]:
	"""
		Convert string to float value and check his range.
	"""
	def float_in_range_impl(x: str) -> float:
		x = float(x)
		if min_ < x < max_ or (include_min and x == min_) or (include_max and x == max_):
			return x
		else:
			raise ValueError("Value \"{:s}\" is not a float in range {:s}{:f},{:f}{:s}".format(
				str(x), "[" if include_min else "]", min_, max_, "]" if include_max else "["))
	return float_in_range_impl


def positive_float():
	return float_in_range(0.0, math.inf)
