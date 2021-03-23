
from hydra.utils import DictConfig
from pytorch_lightning import LightningDataModule
from typing import Callable, Optional

from .ads import ADSPartialDataModule
from .cifar10 import CIFAR10PartialDataModule
from .esc10 import ESC10PartialDataModule
from .gsc import GSCPartialDataModule
from .pvc import PVCPartialDataModule
from .ubs8k import UBS8KPartialDataModule


def get_partial_datamodule_from_cfg(
	cfg: DictConfig,
	transform_train: Optional[Callable],
	transform_val: Optional[Callable],
	target_transform: Optional[Callable],
) -> LightningDataModule:
	"""
		Returns the LightningDataModule corresponding to the config.

		:param cfg: The hydra config.
		:param transform_train: The transform to apply to train data.
		:param transform_val: The transform to apply to validation and test data.
		:param target_transform: The transform to apply to train, validation and test targets.
		:return: The LightningDataModule build from config and transforms.
	"""

	datamodule_params = dict(
		dataset_root=cfg.dataset.root,
		transform_train=transform_train,
		transform_val=transform_val,
		target_transform=target_transform,
		bsize=cfg.bsize,
		num_workers=cfg.cpus,
		drop_last=False,
		pin_memory=False,
	)

	if cfg.dataset.name == "ADS":
		datamodule = ADSPartialDataModule(
			**datamodule_params,
			ratio=cfg.ratio,
			train_subset=cfg.dataset.train_subset,
			nb_train_steps=cfg.dataset.nb_train_steps,
		)
	elif cfg.dataset.name == "CIFAR10":
		datamodule = CIFAR10PartialDataModule(
			**datamodule_params,
			ratio=cfg.ratio,
			download_dataset=cfg.dataset.download,
		)
	elif cfg.dataset.name == "ESC10":
		datamodule = ESC10PartialDataModule(
			**datamodule_params,
			ratio=cfg.ratio,
			download_dataset=cfg.dataset.download,
			folds_train=cfg.dataset.folds_train,
			folds_val=cfg.dataset.folds_val
		)
	elif cfg.dataset.name == "GSC":
		datamodule = GSCPartialDataModule(
			**datamodule_params,
			ratio=cfg.ratio,
			download_dataset=cfg.dataset.download,
		)
	elif cfg.dataset.name == "PVC":
		datamodule = PVCPartialDataModule(
			**datamodule_params,
			ratio=cfg.ratio,
			nb_train_steps=cfg.dataset.nb_train_steps,
		)
	elif cfg.dataset.name == "UBS8K":
		datamodule = UBS8KPartialDataModule(
			**datamodule_params,
			ratio=cfg.ratio,
			folds_train=cfg.dataset.folds_train,
			folds_val=cfg.dataset.folds_val,
		)
	else:
		raise RuntimeError(
			f"Unknown dataset name '{cfg.dataset.name}'. "
			f"Must be one of {('ADS', 'CIFAR10', 'ESC10', 'GSC', 'PVC', 'UBS8K')}."
		)

	return datamodule
