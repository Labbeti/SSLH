
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from typing import Callable, Optional

from mlu.datasets.wrappers import TransformDataset
from sslh.datasets.ads import SingleBalancedSampler, class_balance_split, SingleAudioset


NUM_CLASSES = 527


class ADSPartialDataModule(LightningDataModule):
	def __init__(
		self,
		dataset_root: str,
		transform_train: Optional[Callable] = None,
		transform_val: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		bsize: int = 256,
		num_workers: int = 4,
		drop_last: bool = False,
		pin_memory: bool = False,
		ratio: float = 0.1,
		nb_train_steps: Optional[int] = 125000,
		train_subset: str = "unbalanced",
	):
		"""
			LightningDataModule of AudioSet (ADS) for partial supervised trainings.

			Note: The subset of the dataset has approximately the same class distribution.

			:param dataset_root: The root path of the dataset.
			:param transform_train: The optional transform to apply to train data. (default: None)
			:param transform_val: The optional transform to apply to validation data. (default: None)
			:param target_transform: The optional transform to apply to train and validation targets. (default: None)
			:param bsize: The batch size used for training and validation. (default: 30)
			:param num_workers: The number of workers for each dataloader. (default: 4)
			:param drop_last: If True, drop the last incomplete batch. (default: False)
			:param pin_memory: If True, pin the memory of dataloader. (default: False)
			:param ratio: The ratio of the subset len in [0, 1]. (default: 0.1)
			:param nb_train_steps: The number of train steps for AudioSet.
				If None, the number will be set to the number of train labeled data.
				(default: 125000)
			:param train_subset: The AudioSet train subset to use.
				Can be 'balanced' (~20K samples) or 'unbalanced' (~2M samples).
				(default: 'unbalanced')
		"""
		if train_subset not in ('balanced', 'unbalanced'):
			raise ValueError(f"Train subsets available are {('balanced', 'unbalanced')}.")

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
		self.train_subset = train_subset

		self.train_dataset_raw = None
		self.val_dataset_raw = None
		self.test_dataset_raw = None

		self.rdcc_nbytes = 512 * 1024 ** 2
		self.data_shape = (320000,)  # (64, 500)
		self.data_key = "waveform"  # "data"

		self.sampler = None

	def prepare_data(self, *args, **kwargs):
		pass

	def setup(self, stage: Optional[str] = None):
		self.train_dataset_raw = SingleAudioset(
			root=self.dataset_root,
			transform=None,
			version=self.train_subset,
			rdcc_nbytes=self.rdcc_nbytes,
			data_shape=self.data_shape,
			data_key=self.data_key,
		)

		self.val_dataset_raw = SingleAudioset(
			root=self.dataset_root,
			transform=None,
			version="eval",
			rdcc_nbytes=self.rdcc_nbytes,
			data_shape=self.data_shape,
			data_key=self.data_key,
		)

		self.test_dataset_raw = None

		# Setup split
		indexes_s, _indexes_u = class_balance_split(self.train_dataset_raw, self.ratio, None, verbose=False)

		if self.nb_train_steps is None:
			nb_max_samples_s = len(indexes_s) * self.bsize_train
		else:
			nb_max_samples_s = self.nb_train_steps * self.bsize_train

		self.sampler = SingleBalancedSampler(self.train_dataset_raw, indexes_s, nb_max_samples_s, True)

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
