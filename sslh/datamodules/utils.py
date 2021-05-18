
from typing import List, Tuple, Union


def guess_folds(
	folds_train: Union[List[int], int, None],
	folds_val: Union[List[int], int, None],
	folds: List[int],
) -> Tuple[List[int], List[int]]:
	"""
		Use the folds_train and folds to guess folds_val OR use the folds_val and folds to guess folds_train.

		:param folds_train: The training folds.
		:param folds_val: The validation folds.
		:param folds: The list of folds.
		:return: A tuple of folds (training folds, validation folds).
	"""
	if isinstance(folds_train, int):
		folds_train = [folds_train]
	if isinstance(folds_val, int):
		folds_val = [folds_val]
	folds = set(folds)

	if folds_train is None and folds_val is None:
		folds_val = [len(folds)]
		folds_train = list(folds.difference(folds_val))

	elif folds_train is None:
		folds_train = list(folds.difference(folds_val))

	elif folds_val is None:
		folds_val = list(folds.difference(folds_train))

	return folds_train, folds_val
