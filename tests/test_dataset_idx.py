from unittest import main, TestCase
from ssl.datasets.utils import split_classes_idx


class DatasetIdxTest(TestCase):
	def test_expected(self):
		tests = [
			([[1, 2], [3, 4], [5, 6]], [0.5, 0.5]),
			([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [0.5, 0.25, 0.25]),
		]
		expected_lst = [
			[[1, 3, 5], [2, 4, 6]],
			[[1, 2, 5, 6, 9, 10], [3, 7, 11], [4, 8, 12]],
		]
		for (indices, ratios), expected in zip(tests, expected_lst):
			idx_split = split_classes_idx(indices, ratios)
			self.assertEqual(idx_split, expected)

	def test_reduce(self):
		cls_idx = [[0, 1, 2, 3], [4, 5, 6, 7]]
		expected = [0, 1, 4, 5]
		idx_split = split_classes_idx(cls_idx, [0.5])[0]
		self.assertEqual(idx_split, expected)


if __name__ == "__main__":
	main()
