
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Callable, List, Optional, Tuple

from mlu.datasets.split.monolabel import balanced_split
from mlu.datasets.wrappers import TransformDataset, NoLabelDataset
from sslh.datamodules.utils import guess_folds
from sslh.datasets.esc10 import ESC10


N_CLASSES = 10
FOLDS = [1, 2, 3, 4, 5]


class ESC10DataModuleSSL(LightningDataModule):
	def __init__(
		self,
		root: str,
		transform_train_s: Optional[Callable] = None,
		transform_train_u: Optional[Callable] = None,
		transform_val: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		bsize_train_s: int = 30,
		bsize_train_u: int = 30,
		n_workers_s: int = 2,
		n_workers_u: int = 3,
		drop_last: bool = True,
		pin_memory: bool = False,
		ratio_s: float = 0.1,
		ratio_u: float = 0.9,
		duplicate_loader_s: bool = False,
		download_dataset: bool = True,
		folds_train: Optional[List[int]] = None,
		folds_val: Optional[List[int]] = None,
	):
		"""
			LightningDataModule of ESC-10 for semi-supervised trainings.

			Note: The splits of the dataset has the same class distribution.

			:param root: The root path of the dataset.
			:param transform_train_s: The optional transform to apply to supervised train data. (default: None)
			:param transform_train_u: The optional transform to apply to unsupervised train data. (default: None)
			:param transform_val: The optional transform to apply to validation data. (default: None)
			:param target_transform: The optional transform to apply to train and validation targets. (default: None)
			:param bsize_train_s: The batch size used for supervised train data. (default: 64)
			:param bsize_train_u: The batch size used for unsupervised train data. (default: 64)
			:param n_workers_s: The number of workers for supervised train dataloader. (default: 2)
			:param n_workers_s: The number of workers for unsupervised train dataloader. (default: 3)
			:param drop_last: If True, drop the last incomplete batch. (default: False)
			:param pin_memory: If True, pin the memory of dataloader. (default: False)
			:param ratio_s: The ratio of the supervised subset len in [0, 1]. (default: 0.1)
			:param ratio_u: The ratio of the unsupervised subset len in [0, 1]. (default: 0.9)
			:param duplicate_loader_s: If True, duplicate the supervised dataloader for DCT training. (default: False)
			:param download_dataset: If True, automatically download the dataset in the root directory. (default: True)
			:param folds_train: The folds used for training.
				If None and folds_val is not None, then use the unused folds of validation.
				If both folds_train and folds_val are None, then the default folds are used:
					[1, 2, 3, 4] for folds_train and [5] for folds_val.
				(default: None)
			:param folds_val: The folds used for validation.
				If None and folds_train is not None, then use the unused folds of training.
				If both folds_train and folds_val are None, then the default folds are used:
					[1, 2, 3, 4] for folds_train and [5] for folds_val.
				(default: None)
		"""
		super().__init__()
		self.root = root
		self.transform_train_s = transform_train_s
		self.transform_train_u = transform_train_u
		self.transform_val = transform_val
		self.transform_test = transform_val
		self.target_transform = target_transform
		self.bsize_train_s = bsize_train_s
		self.bsize_train_u = bsize_train_u
		self.bsize_val = bsize_train_s + bsize_train_u
		self.bsize_test = bsize_train_s + bsize_train_u
		self.n_workers_s = n_workers_s
		self.n_workers_u = n_workers_u
		self.drop_last = drop_last
		self.pin_memory = pin_memory
		self.ratio_s = ratio_s
		self.ratio_u = ratio_u
		self.duplicate_loader_s = duplicate_loader_s

		self.download_dataset = download_dataset
		self.folds_train, self.folds_val = guess_folds(folds_train, folds_val, FOLDS)

		self.train_dataset_raw = None
		self.val_dataset_raw = None
		self.test_dataset_raw = None

		self.sampler_s = None
		self.sampler_u = None
		self.example_input_array = None

	def prepare_data(self, *args, **kwargs):
		if self.download_dataset:
			_ = ESC10(self.root, folds=tuple(FOLDS), download=True)

	def setup(self, stage: Optional[str] = None):
		if stage == 'fit':
			self.train_dataset_raw = ESC10(root=self.root, folds=tuple(self.folds_train), download=False)
			self.val_dataset_raw = ESC10(root=self.root, folds=tuple(self.folds_val), download=False)

			# Setup split
			ratios = [self.ratio_s, self.ratio_u]
			indexes_s, indexes_u = balanced_split(
				dataset=self.train_dataset_raw,
				n_classes=N_CLASSES,
				ratios=ratios,
				target_one_hot=False,
			)
			self.sampler_s = SubsetRandomSampler(indexes_s)
			self.sampler_u = SubsetRandomSampler(indexes_u)

			dataloader = self.val_dataloader()
			xs, ys = next(iter(dataloader))
			self.example_input_array = xs
			self.dims = tuple(xs.shape)

		elif stage == 'test':
			self.test_dataset_raw = None

	def train_dataloader(self) -> Tuple[DataLoader, ...]:
		train_dataset_s = TransformDataset(self.train_dataset_raw, self.transform_train_s, index=0)
		train_dataset_s = TransformDataset(train_dataset_s, self.target_transform, index=1)

		train_dataset_u = TransformDataset(self.train_dataset_raw, self.transform_train_u, index=0)
		train_dataset_u = NoLabelDataset(train_dataset_u)

		loader_s = DataLoader(
			dataset=train_dataset_s,
			batch_size=self.bsize_train_s,
			num_workers=self.n_workers_s,
			sampler=self.sampler_s,
			drop_last=self.drop_last,
			pin_memory=self.pin_memory,
		)
		loader_u = DataLoader(
			dataset=train_dataset_u,
			batch_size=self.bsize_train_u,
			num_workers=self.n_workers_u,
			sampler=self.sampler_u,
			drop_last=self.drop_last,
			pin_memory=self.pin_memory,
		)

		if not self.duplicate_loader_s:
			loaders = loader_s, loader_u
		else:
			loaders = loader_s, loader_s, loader_u

		return loaders

	def val_dataloader(self) -> Optional[DataLoader]:
		val_dataset = self.val_dataset_raw
		if val_dataset is None:
			return None

		val_dataset = TransformDataset(val_dataset, self.transform_val, index=0)
		val_dataset = TransformDataset(val_dataset, self.target_transform, index=1)

		loader = DataLoader(
			dataset=val_dataset,
			batch_size=self.bsize_val,
			num_workers=self.n_workers_s + self.n_workers_u,
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
			num_workers=self.n_workers_s + self.n_workers_u,
			drop_last=False,
		)
		return loader
