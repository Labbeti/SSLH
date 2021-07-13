
from hydra.utils import DictConfig
from pytorch_lightning import LightningDataModule
from typing import Callable, Optional

from .ads import ADSDataModuleSup
from .cifar10 import CIFAR10DataModuleSup
from .esc10 import ESC10DataModuleSup
from .fsd50k import FSD50KDataModuleSup
from .gsc import GSCDataModuleSup
from .pvc import PVCDataModuleSup
from .ubs8k import UBS8KDataModuleSup


def get_datamodule_sup_from_cfg(
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
		root=cfg.data.root,
		transform_train=transform_train,
		transform_val=transform_val,
		target_transform=target_transform,
		bsize=cfg.bsize,
		n_workers=cfg.cpus,
		drop_last=False,
		pin_memory=False,
		ratio=cfg.ratio,
	)

	if cfg.data.acronym == 'ADS':
		datamodule = ADSDataModuleSup(
			**datamodule_params,
			train_subset=cfg.data.train_subset,
			n_train_steps=cfg.data.n_train_steps,
			sampler_s_balanced=cfg.data.sampler_s_balanced,
			pre_computed_specs=cfg.data.pre_computed_specs,
		)
	elif cfg.data.acronym == 'CIFAR10':
		datamodule = CIFAR10DataModuleSup(
			**datamodule_params,
			download_dataset=cfg.data.download,
		)
	elif cfg.data.acronym == 'ESC10':
		datamodule = ESC10DataModuleSup(
			**datamodule_params,
			download_dataset=cfg.data.download,
			folds_train=cfg.data.folds_train,
			folds_val=cfg.data.folds_val
		)
	elif cfg.data.acronym == 'FSD50K':
		datamodule = FSD50KDataModuleSup(
			**datamodule_params,
			download_dataset=cfg.data.download,
			n_train_steps=cfg.data.n_train_steps,
			sampler_s_balanced=cfg.data.sampler_s_balanced,
		)
	elif cfg.data.acronym == 'GSC':
		datamodule = GSCDataModuleSup(
			**datamodule_params,
			download_dataset=cfg.data.download,
		)
	elif cfg.data.acronym == 'PVC':
		datamodule = PVCDataModuleSup(
			**datamodule_params,
			n_train_steps=cfg.data.n_train_steps,
		)
	elif cfg.data.acronym == 'UBS8K':
		datamodule = UBS8KDataModuleSup(
			**datamodule_params,
			folds_train=cfg.data.folds_train,
			folds_val=cfg.data.folds_val,
		)
	else:
		raise RuntimeError(
			f'Unknown dataset name "{cfg.data.acronym}". '
			f'Must be one of {("ADS", "CIFAR10", "ESC10", "FSD50K", "GSC", "PVC", "UBS8K")}.'
		)

	return datamodule
