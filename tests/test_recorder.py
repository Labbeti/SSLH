
import numpy as np

from sslh.utils.recorder.recorder import Recorder
from unittest import TestCase, main


class TestRecorder(TestCase):
	def test_means(self):
		recorder = Recorder()

		test_set = {"a": [1, 2, 3, 4], "b": [0, -1]}
		for name, values in test_set.items():
			for v in values:
				recorder.add_scalar(name, v)

		for n, mean in recorder.get_current_means().items():
			expected = np.mean(test_set[n]).item()
			self.assertAlmostEqual(mean, expected)


if __name__ == "__main__":
	main()
