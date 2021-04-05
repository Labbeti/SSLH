
import os.path as osp

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from typing import Callable, List, Optional

from mlu.datasets.wrappers import TransformDataset
from mlu.datasets.utils import generate_samplers_split
from sslh.datamodules.utils import guess_folds
from sslh.datasets.ubs8k_ import UBS8KDataset
from ubs8k.datasetManager import DatasetManager as UBS8KDatasetManager


NUM_CLASSES = 10
FOLDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class UBS8KPartialDataModule(LightningDataModule):
	def __init__(
		self,
		dataset_root: str,
		transform_train: Optional[Callable] = None,
		transform_val: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		bsize: int = 30,
		num_workers: int = 4,
		drop_last: bool = False,
		pin_memory: bool = False,
		ratio: float = 0.1,
		folds_train: Optional[List[int]] = None,
		folds_val: Optional[List[int]] = None,
	):
		"""
			LightningDataModule of UrbanSound8K (UBS8K) for partial supervised trainings.

			Note: The subset of the dataset has the same class distribution.

			:param dataset_root: The root path of the dataset.
			:param transform_train: The optional transform to apply to train data. (default: None)
			:param transform_val: The optional transform to apply to validation data. (default: None)
			:param target_transform: The optional transform to apply to train and validation targets. (default: None)
			:param bsize: The batch size used for training and validation. (default: 30)
			:param num_workers: The number of workers for each dataloader. (default: 4)
			:param drop_last: If True, drop the last incomplete batch. (default: False)
			:param pin_memory: If True, pin the memory of dataloader. (default: False)
			:param ratio: The ratio of the subset len in [0, 1]. (default: 0.1)
			:param folds_train: The folds used for training.
				If None and folds_val is not None, then use the unused folds of validation.
				If both folds_train and folds_val are None, then the default folds are used:
					[1, 2, 3, 4, 5, 6, 7, 8, 9] for folds_train and [10] for folds_val.
				(default: None)
			:param folds_val: The folds used for validation.
				If None and folds_train is not None, then use the unused folds of training.
				If both folds_train and folds_val are None, then the default folds are used:
					[1, 2, 3, 4, 5, 6, 7, 8, 9] for folds_train and [10] for folds_val.
				(default: None)
		"""
		super().__init__()
		self.dataset_root = dataset_root
		self.transform_train = transform_train
		self.transform_val = transform_val
		self.transform_test = transform_val
		self.target_transform = target_transform
		self.bsize_train = bsize
		self.bsize_val = bsize
		self.bsize_test = bsize
		self.num_workers = num_workers
		self.drop_last = drop_last
		self.pin_memory = pin_memory
		self.ratio = ratio

		self.folds_train, self.folds_val = guess_folds(folds_train, folds_val, FOLDS)

		self.train_dataset_raw = None
		self.val_dataset_raw = None
		self.test_dataset_raw = None

		self.sampler = None
		self.manager = None

	def prepare_data(self, *args, **kwargs):
		pass

	def setup(self, stage: Optional[str] = None):
		if not osp.isdir(self.dataset_root):
			raise RuntimeError(f"Unknown dataset root dirpath '{self.dataset_root}' for UBS8K.")

		metadata_root = osp.join(self.dataset_root, "metadata")
		audio_root = osp.join(self.dataset_root, "audio")

		if not osp.isdir(metadata_root):
			raise RuntimeError(f"Unknown metadata root dirpath '{metadata_root}' for UBS8K.")
		if not osp.isdir(audio_root):
			raise RuntimeError(f"Unknown audio root dirpath '{audio_root}' for UBS8K.")

		self.manager = UBS8KDatasetManager(metadata_root, audio_root)

		self.train_dataset_raw = UBS8KDataset(self.manager, folds=self.folds_train)
		self.val_dataset_raw = UBS8KDataset(self.manager, folds=self.folds_val)
		self.test_dataset_raw = None

		# Setup split
		ratios = [self.ratio]
		self.sampler = generate_samplers_split(
			dataset=self.train_dataset_raw,
			num_classes=NUM_CLASSES,
			ratios=ratios,
			target_one_hot=False,
		)[0]

	def train_dataloader(self) -> DataLoader:
		train_dataset = self.train_dataset_raw
		train_dataset = TransformDataset(train_dataset, self.transform_val, index=0)
		train_dataset = TransformDataset(train_dataset, self.target_transform, index=1)

		loader = DataLoader(
			dataset=train_dataset,
			batch_size=self.bsize_train,
			num_workers=self.num_workers,
			drop_last=self.drop_last,
			pin_memory=self.pin_memory,
			sampler=self.sampler,
		)
		return loader

	def val_dataloader(self) -> Optional[DataLoader]:
		val_dataset = self.val_dataset_raw
		if val_dataset is None:
			return None

		val_dataset = TransformDataset(val_dataset, self.transform_val, index=0)
		val_dataset = TransformDataset(val_dataset, self.target_transform, index=1)

		loader = DataLoader(
			dataset=val_dataset,
			batch_size=self.bsize_val,
			num_workers=self.num_workers,
			drop_last=False,
		)
		return loader

	def test_dataloader(self) -> Optional[DataLoader]:
		test_dataset = self.test_dataset_raw
		if test_dataset is None:
			return None

		test_dataset = TransformDataset(test_dataset, self.transform_test, index=0)
		test_dataset = TransformDataset(test_dataset, self.target_transform, index=1)

		loader = DataLoader(
			dataset=test_dataset,
			batch_size=self.bsize_test,
			num_workers=self.num_workers,
			drop_last=False,
		)
		return loader

