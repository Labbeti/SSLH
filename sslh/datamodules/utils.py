
from typing import List, Optional, Tuple


def guess_folds(
	folds_train: Optional[List[int]],
	folds_val: Optional[List[int]],
	folds: List[int]
) -> Tuple[List[int], List[int]]:
	"""
		Use the folds_train and folds to guess folds_val OR use the folds_val and folds to guess folds_train.

		:param folds_train: The training folds.
		:param folds_val: The validation folds.
		:param folds: The list of folds.
		:return: A tuple of folds (training folds, validation folds).
	"""
	folds = set(folds)

	if folds_train is None and folds_val is None:
		folds_val = [len(folds)]
		folds_train = list(folds.difference(folds_val))
	elif folds_train is None:
		folds_train = list(folds.difference(folds_val))
	elif folds_val is None:
		folds_val = list(folds.difference(folds_train))

	return folds_train, folds_val
