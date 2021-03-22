"""
	UrbanSound8k (UBS8K) dataset wrapper.
"""

import numpy as np

from typing import Callable, Sequence, Optional

from ubs8k.datasets import Dataset as UBS8KDatasetBase
from ubs8k.datasetManager import DatasetManager as UBS8KDatasetManager


class UBS8KDataset(UBS8KDatasetBase):
	"""
		UBS8K dataset without pad, crop and cache.
	"""
	def __init__(
		self, manager: UBS8KDatasetManager,
		folds: Sequence[int],
		transform: Optional[Callable] = None,
		cached: bool = False,
	):
		if transform is None:
			augments = ()
		else:
			augments = (transform, )

		super().__init__(
			manager=manager,
			folds=tuple(folds),
			augments=augments,
			cached=cached,
			augment_choser=lambda x: x,
		)

	def _pad_and_crop(self, raw_data):
		return raw_data

	def _generate_data(self, index: int):
		# load the raw_audio
		filename = self.filenames[index]
		raw_audio = self.x[filename]

		# recover ground truth
		y = self.y.at[filename, "classID"]

		# check if augmentation should be applied
		apply_augmentation = self.augment_S if index in self.s_idx else self.augment_U

		# chose augmentation, if no return an empty list
		augment_fn = self.augment_choser(self.augments) if self.augments else []

		# Apply augmentation, only one that applies on the signal will be executed
		audio_transformed, cache_id = self._apply_augmentation(raw_audio, augment_fn, filename, apply_augmentation)
		y = np.asarray(y)

		# call end of generation callbacks
		self.end_of_generation_callback()

		return audio_transformed, y
