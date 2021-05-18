
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
		dataset_root=cfg.dataset.root,
		transform_train=transform_train,
		transform_val=transform_val,
		target_transform=target_transform,
		bsize=cfg.bsize,
		n_workers=cfg.cpus,
		drop_last=False,
		pin_memory=False,
		ratio=cfg.ratio,
	)

	if cfg.dataset.acronym == 'ADS':
		datamodule = ADSDataModuleSup(
			**datamodule_params,
			train_subset=cfg.dataset.train_subset,
			n_train_steps=cfg.dataset.n_train_steps,
			sampler_s_balanced=cfg.dataset.sampler_s_balanced,
			pre_computed_specs=cfg.dataset.pre_computed_specs,
		)
	elif cfg.dataset.acronym == 'CIFAR10':
		datamodule = CIFAR10DataModuleSup(
			**datamodule_params,
			download_dataset=cfg.dataset.download,
		)
	elif cfg.dataset.acronym == 'ESC10':
		datamodule = ESC10DataModuleSup(
			**datamodule_params,
			download_dataset=cfg.dataset.download,
			folds_train=cfg.dataset.folds_train,
			folds_val=cfg.dataset.folds_val
		)
	elif cfg.dataset.acronym == 'FSD50K':
		datamodule = FSD50KDataModuleSup(
			**datamodule_params,
			download_dataset=cfg.dataset.download,
			n_train_steps=cfg.dataset.n_train_steps,
			sampler_s_balanced=cfg.dataset.sampler_s_balanced,
		)
	elif cfg.dataset.acronym == 'GSC':
		datamodule = GSCDataModuleSup(
			**datamodule_params,
			download_dataset=cfg.dataset.download,
		)
	elif cfg.dataset.acronym == 'PVC':
		datamodule = PVCDataModuleSup(
			**datamodule_params,
			n_train_steps=cfg.dataset.n_train_steps,
		)
	elif cfg.dataset.acronym == 'UBS8K':
		datamodule = UBS8KDataModuleSup(
			**datamodule_params,
			folds_train=cfg.dataset.folds_train,
			folds_val=cfg.dataset.folds_val,
		)
	else:
		raise RuntimeError(
			f'Unknown dataset name "{cfg.dataset.acronym}". '
			f'Must be one of {("ADS", "CIFAR10", "ESC10", "FSD50K", "GSC", "PVC", "UBS8K")}.'
		)

	return datamodule
