
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from typing import Callable, Optional, Tuple

from mlu.datasets.utils import generate_samplers_split
from mlu.datasets.wrappers import TransformDataset, NoLabelDataset
from sslh.datasets.gsc import SpeechCommands


NUM_CLASSES = 35


class GSCSemiDataModule(LightningDataModule):
	def __init__(
		self,
		dataset_root: str,
		transform_train_s: Optional[Callable] = None,
		transform_train_u: Optional[Callable] = None,
		transform_val: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		bsize_train_s: int = 64,
		bsize_train_u: int = 64,
		num_workers_s: int = 2,
		num_workers_u: int = 3,
		drop_last: bool = True,
		pin_memory: bool = False,
		ratio_s: float = 0.1,
		ratio_u: float = 0.9,
		download_dataset: bool = True,
	):
		super().__init__()
		self.dataset_root = dataset_root
		self.transform_train_s = transform_train_s
		self.transform_train_u = transform_train_u
		self.transform_val = transform_val
		self.transform_test = transform_val
		self.target_transform = target_transform
		self.bsize_train_s = bsize_train_s
		self.bsize_train_u = bsize_train_u
		self.bsize_val = bsize_train_s + bsize_train_u
		self.bsize_test = bsize_train_s + bsize_train_u
		self.num_workers_s = num_workers_s
		self.num_workers_u = num_workers_u
		self.drop_last = drop_last
		self.pin_memory = pin_memory
		self.ratio_s = ratio_s
		self.ratio_u = ratio_u

		self.download_dataset = download_dataset

		self.train_dataset_raw = None
		self.val_dataset_raw = None
		self.test_dataset_raw = None

		self.sampler_s = None
		self.sampler_u = None

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
		ratios = [self.ratio_s, self.ratio_u]
		self.sampler_s, self.sampler_u = generate_samplers_split(
			dataset=self.train_dataset_raw,
			num_classes=NUM_CLASSES,
			ratios=ratios,
			target_one_hot=False,
		)

	def train_dataloader(self) -> Tuple[DataLoader, DataLoader]:
		train_dataset_s = TransformDataset(self.train_dataset_raw, self.transform_train_s, index=0)
		train_dataset_s = TransformDataset(train_dataset_s, self.target_transform, index=1)

		train_dataset_u = TransformDataset(self.train_dataset_raw, self.transform_train_u, index=0)
		train_dataset_u = NoLabelDataset(train_dataset_u)

		loader_s = DataLoader(
			dataset=train_dataset_s,
			batch_size=self.bsize_train_s,
			num_workers=self.num_workers_s,
			sampler=self.sampler_s,
			drop_last=self.drop_last,
			pin_memory=self.pin_memory,
		)
		loader_u = DataLoader(
			dataset=train_dataset_u,
			batch_size=self.bsize_train_u,
			num_workers=self.num_workers_u,
			sampler=self.sampler_u,
			drop_last=self.drop_last,
			pin_memory=self.pin_memory,
		)
		return loader_s, loader_u

	def val_dataloader(self) -> Optional[DataLoader]:
		val_dataset = self.val_dataset_raw
		if val_dataset is None:
			return None

		val_dataset = TransformDataset(val_dataset, self.transform_val, index=0)
		val_dataset = TransformDataset(val_dataset, self.target_transform, index=1)

		loader = DataLoader(
			dataset=val_dataset,
			batch_size=self.bsize_val,
			num_workers=self.num_workers_s + self.num_workers_u,
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
			num_workers=self.num_workers_s + self.num_workers_u,
			drop_last=False,
		)
		return loader
