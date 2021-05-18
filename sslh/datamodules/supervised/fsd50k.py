
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from typing import Callable, Optional

from mlu.datasets.fsd50k import FSD50K, FSD50KSubset
from mlu.datasets.samplers import BalancedSampler, SubsetCycleSampler
from mlu.datasets.split.multilabel import balanced_split, get_indexes_per_class
from mlu.datasets.wrappers import TransformDataset


N_CLASSES = 200


class FSD50KDataModuleSup(LightningDataModule):
	def __init__(
		self,
		dataset_root: str,
		transform_train: Optional[Callable] = None,
		transform_val: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		bsize: int = 30,
		n_workers: int = 4,
		drop_last: bool = False,
		pin_memory: bool = False,
		ratio: float = 1.0,
		download_dataset: bool = False,
		n_train_steps: Optional[int] = 1000,
		sampler_s_balanced: bool = True,
	):
		"""
			LightningDataModule of FSD50K (FSD50K) for partial supervised trainings.

			:param dataset_root: The root path of the dataset.
			:param transform_train: The optional transform to apply to train data. (default: None)
			:param transform_val: The optional transform to apply to validation data. (default: None)
			:param target_transform: The optional transform to apply to train and validation targets. (default: None)
			:param bsize: The batch size used for training and validation. (default: 30)
			:param n_workers: The number of workers for each dataloader. (default: 4)
			:param drop_last: If True, drop the last incomplete batch. (default: False)
			:param pin_memory: If True, pin the memory of dataloader. (default: False)
			:param download_dataset: TODO
			:param n_train_steps: The number of train steps for AudioSet.
				If None, the number will be set to the number of train labeled data.
				(default: 1000)
			:param sampler_s_balanced: TODO
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
		self.n_workers = n_workers
		self.drop_last = drop_last
		self.pin_memory = pin_memory
		self.ratio = ratio

		self.download_dataset = download_dataset
		self.n_train_steps = n_train_steps
		self.sampler_s_balanced = sampler_s_balanced

		self.train_dataset_raw = None
		self.val_dataset_raw = None
		self.test_dataset_raw = None

		self.sampler_s = None
		self.example_input_array = None

	def prepare_data(self, *args, **kwargs):
		if self.download_dataset:
			_ = FSD50K(root=self.dataset_root, subset=FSD50KSubset.DEV, download=True)

	def setup(self, stage: Optional[str] = None):
		dataset_params = dict(
			root=self.dataset_root,
			download=False,
		)
		if stage == 'fit':
			self.train_dataset_raw = FSD50K(subset=FSD50KSubset.TRAIN, **dataset_params)
			self.val_dataset_raw = FSD50K(subset=FSD50KSubset.VAL, **dataset_params)

			if self.ratio >= 1.0:
				indexes_s = list(range(len(self.train_dataset_raw)))
			else:
				indexes_s, _indexes_u = balanced_split(
					dataset=self.train_dataset_raw,
					n_classes=N_CLASSES,
					target_type='indexes',
					ratio_s=self.ratio,
					ratio_u=None,
				)

			if self.n_train_steps is None:
				n_train_samples_s = len(indexes_s)
			else:
				n_train_samples_s = self.n_train_steps * self.bsize_train

			if self.sampler_s_balanced:
				indexes_per_class = get_indexes_per_class(
					dataset=self.train_dataset_raw,
					n_classes=N_CLASSES,
					target_type='indexes',
					indexes=indexes_s,
				)
				self.sampler_s = BalancedSampler(indexes_per_class, n_train_samples_s)
			else:
				self.sampler_s = SubsetCycleSampler(indexes_s, n_train_samples_s)

			dataloader = self.val_dataloader()
			xs, ys = next(iter(dataloader))
			self.example_input_array = xs
			self.dims = tuple(xs.shape)

		elif stage == 'test':
			self.test_dataset_raw = FSD50K(subset=FSD50KSubset.EVAL, **dataset_params)

	def train_dataloader(self) -> DataLoader:
		train_dataset = self.train_dataset_raw
		train_dataset = TransformDataset(train_dataset, self.transform_train, index=0)
		train_dataset = TransformDataset(train_dataset, self.target_transform, index=1)

		loader = DataLoader(
			dataset=train_dataset,
			batch_size=self.bsize_train,
			num_workers=self.n_workers,
			drop_last=self.drop_last,
			pin_memory=self.pin_memory,
			sampler=self.sampler_s,
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
