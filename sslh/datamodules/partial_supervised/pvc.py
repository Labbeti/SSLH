
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from typing import Callable, Optional

from mlu.datasets.wrappers import TransformDataset
from sslh.datasets.pvc import ComParE2021PRS, IterationBalancedSampler, class_balance_split


NUM_CLASSES = 5


class PVCPartialDataModule(LightningDataModule):
	def __init__(
		self,
		dataset_root: str,
		transform_train: Optional[Callable] = None,
		transform_val: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		bsize: int = 256,
		num_workers: int = 5,
		drop_last: bool = False,
		pin_memory: bool = False,
		ratio: float = 0.1,
		nb_train_steps: Optional[int] = None,
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
		self.ratio = ratio

		self.nb_train_steps = nb_train_steps

		self.train_dataset_raw = None
		self.val_dataset_raw = None
		self.test_dataset_raw = None

		self.sampler = None

	def prepare_data(self, *args, **kwargs):
		pass

	def setup(self, stage: Optional[str] = None):
		self.train_dataset_raw = ComParE2021PRS(self.dataset_root, "train", transform=None)
		self.val_dataset_raw = ComParE2021PRS(self.dataset_root, "devel", transform=None)
		# The "test" subset is unlabeled, so we do not use it for now
		self.test_dataset_raw = None

		indexes_s, _indexes_u = class_balance_split(self.train_dataset_raw, self.ratio, None)

		if self.nb_train_steps is None:
			nb_max_samples_s = len(indexes_s) * self.bsize_train
		else:
			nb_max_samples_s = self.nb_train_steps * self.bsize_train

		self.sampler = IterationBalancedSampler(self.train_dataset_raw, indexes_s, nb_max_samples_s)

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
