
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from typing import Callable, Optional, Tuple

from mlu.datasets.samplers import SubsetCycleSampler
from mlu.datasets.wrappers import TransformDataset, NoLabelDataset
from sslh.datasets.pvc import ComParE2021PRS, IterationBalancedSampler, class_balance_split


NUM_CLASSES = 5


class PVCSemiDataModule(LightningDataModule):
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
		nb_train_steps_u: Optional[int] = None,
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

		self.nb_train_steps_u = nb_train_steps_u

		self.train_dataset_raw = None
		self.val_dataset_raw = None
		self.test_dataset_raw = None

		self.sampler_s = None
		self.sampler_u = None

	def prepare_data(self, *args, **kwargs):
		pass

	def setup(self, stage: Optional[str] = None):
		self.train_dataset_raw = ComParE2021PRS(self.dataset_root, "train", transform=None)
		self.val_dataset_raw = ComParE2021PRS(self.dataset_root, "devel", transform=None)
		# The "test" subset is unlabeled, so we do not use it for now
		self.test_dataset_raw = None

		indexes_s, indexes_u = class_balance_split(self.train_dataset_raw, self.ratio_s, self.ratio_u)

		if self.nb_train_steps_u is None:
			nb_train_samples_s = len(indexes_s) * self.bsize_train_s
			nb_train_samples_u = len(indexes_u) * self.bsize_train_u
		else:
			nb_train_samples_s = self.nb_train_steps_u * self.bsize_train_s
			nb_train_samples_u = self.nb_train_steps_u * self.bsize_train_u

		self.sampler_s = IterationBalancedSampler(self.train_dataset_raw, indexes_s, nb_train_samples_s)
		self.sampler_u = SubsetCycleSampler(indexes_u, nb_train_samples_u)

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
