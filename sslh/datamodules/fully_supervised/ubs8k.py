
import os.path as osp

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from typing import Callable, List, Optional

from mlu.datasets.wrappers import TransformDataset
from sslh.datamodules.utils import guess_folds
from sslh.datasets.ubs8k_ import UBS8KDataset
from ubs8k.datasetManager import DatasetManager as UBS8KDatasetManager


NUM_CLASSES = 10
FOLDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class UBS8KFullyDataModule(LightningDataModule):
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

		self.folds_train, self.folds_val = guess_folds(folds_train, folds_val, FOLDS)

		self.train_dataset_raw = None
		self.val_dataset_raw = None
		self.test_dataset_raw = None

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
