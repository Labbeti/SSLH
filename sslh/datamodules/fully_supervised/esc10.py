
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from typing import Callable, List, Optional

from mlu.datasets.wrappers import TransformDataset
from sslh.datamodules.utils import guess_folds
from sslh.datasets.esc10 import ESC10


NUM_CLASSES = 10
FOLDS = [1, 2, 3, 4, 5]


class ESC10FullyDataModule(LightningDataModule):
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
		download_dataset: bool = True,
		folds_train: Optional[List[int]] = None,
		folds_val: Optional[List[int]] = None,
	):
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

		self.download_dataset = download_dataset
		self.folds_train, self.folds_val = guess_folds(folds_train, folds_val, FOLDS)

		self.train_dataset_raw = None
		self.val_dataset_raw = None
		self.test_dataset_raw = None

	def prepare_data(self, *args, **kwargs):
		if self.download_dataset:
			_ = ESC10(self.dataset_root, folds=tuple(self.folds_train), download=True)
			_ = ESC10(self.dataset_root, folds=tuple(self.folds_val), download=True)

	def setup(self, stage: Optional[str] = None):
		self.train_dataset_raw = ESC10(self.dataset_root, folds=tuple(self.folds_train), download=False, transform=None)
		self.val_dataset_raw = ESC10(self.dataset_root, folds=tuple(self.folds_val), download=False, transform=None)
		self.test_dataset_raw = None

	def train_dataloader(self) -> DataLoader:
		train_dataset = self.train_dataset_raw
		train_dataset = TransformDataset(train_dataset, self.transform_train, index=0)
		train_dataset = TransformDataset(train_dataset, self.target_transform, index=1)

		loader = DataLoader(
			dataset=train_dataset,
			batch_size=self.bsize_train,
			num_workers=self.num_workers,
			shuffle=True,
			drop_last=self.drop_last,
			pin_memory=self.pin_memory,
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
			num_workers=0,
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
			num_workers=0,
			drop_last=False,
		)
		return loader
