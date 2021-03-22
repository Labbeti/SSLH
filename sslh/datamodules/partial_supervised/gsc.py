
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from typing import Callable, Optional

from mlu.datasets.wrappers import TransformDataset
from mlu.datasets.utils import generate_samplers_split
from sslh.datasets.gsc import SpeechCommands


NUM_CLASSES = 35


class GSCPartialDataModule(LightningDataModule):
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
		download_dataset: bool = True,
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

		self.download_dataset = download_dataset

		self.train_dataset_raw = None
		self.val_dataset_raw = None
		self.test_dataset_raw = None

		self.sampler = None

	def prepare_data(self, *args, **kwargs):
		if self.download_dataset:
			_ = SpeechCommands(self.dataset_root, "train", download=True, transform=None)
			_ = SpeechCommands(self.dataset_root, "validation", download=True, transform=None)
			_ = SpeechCommands(self.dataset_root, "testing", download=True, transform=None)

	def setup(self, stage: Optional[str] = None):
		self.train_dataset_raw = SpeechCommands(self.dataset_root, "train", download=False, transform=None)
		self.val_dataset_raw = SpeechCommands(self.dataset_root, "validation", download=False, transform=None)
		self.test_dataset_raw = SpeechCommands(self.dataset_root, "testing", download=False, transform=None)

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
