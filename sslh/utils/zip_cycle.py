
from typing import Iterable, Sized


class ZipCycle(Iterable, Sized):
	"""
		Zip through a list of iterables and sized objects of different lengths.
		Reset the iterators when there and finish iteration when the longest one is over.

		Example :
		r1 = range(1, 4)
		r2 = range(1, 6)
		iters = ZipCycle([r1, r2])
		for v1, v2 in iters:
			print(v1, v2)

		will print :
		1 1
		2 2
		3 3
		1 4
		2 5
	"""

	def __init__(self, iterables: list):
		for iterable in iterables:
			if len(iterable) == 0:
				raise RuntimeError("An iterable is empty.")

		self._iterables = iterables
		self._len = max([len(iterable) for iterable in self._iterables])

	def __iter__(self) -> list:
		cur_iters = [iter(iterable) for iterable in self._iterables]
		cur_count = [0 for _ in self._iterables]

		for _ in range(len(self)):
			items = []

			for i, _ in enumerate(cur_iters):
				if cur_count[i] < len(self._iterables[i]):
					item = next(cur_iters[i])
					cur_count[i] += 1
				else:
					cur_iters[i] = iter(self._iterables[i])
					item = next(cur_iters[i])
					cur_count[i] = 1
				items.append(item)

			yield items

	def __len__(self) -> int:
		return self._len
