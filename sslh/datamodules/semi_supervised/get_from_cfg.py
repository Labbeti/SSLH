
from hydra.utils import DictConfig
from pytorch_lightning import LightningDataModule
from typing import Callable, Optional

from .ads import ADSDataModuleSSL
from .cifar10 import CIFAR10DataModuleSSL
from .esc10 import ESC10DataModuleSSL
from .fsd50k import FSD50KDataModuleSSL
from .gsc import GSCDataModuleSSL
from .pvc import PVCDataModuleSSL
from .ubs8k import UBS8KDataModuleSSL


def get_datamodule_ssl_from_cfg(
	cfg: DictConfig,
	transform_train_s: Optional[Callable],
	transform_train_u: Optional[Callable],
	transform_val: Optional[Callable],
	target_transform: Optional[Callable],
) -> LightningDataModule:
	"""
		Returns the LightningDataModule corresponding to the config.

		:param cfg: The hydra config.
		:param transform_train_s: The transform to apply to train supervised (labeled) data.
		:param transform_train_u: The transform to apply to train unsupervised (unlabeled) data.
		:param transform_val: The transform to apply to validation and test data.
		:param target_transform: The transform to apply to train, validation and test targets.
		:return: The LightningDataModule build from config and transforms.
	"""

	duplicate_loader_s = cfg.expt.duplicate_loader_s if hasattr(cfg.expt, 'duplicate_loader_s') else False

	datamodule_params = dict(
		dataset_root=cfg.dataset.root,
		ratio_s=cfg.ratio_s,
		ratio_u=cfg.ratio_u,
		transform_train_s=transform_train_s,
		transform_train_u=transform_train_u,
		transform_val=transform_val,
		target_transform=target_transform,
		bsize_train_s=cfg.bsize_s,
		bsize_train_u=cfg.bsize_u,
		n_workers_s=round(cfg.cpus / 2),
		n_workers_u=round(cfg.cpus / 2),
		duplicate_loader_s=duplicate_loader_s,
	)

	if cfg.dataset.acronym == 'ADS':
		datamodule = ADSDataModuleSSL(
			**datamodule_params,
			train_subset=cfg.dataset.train_subset,
			n_train_steps=cfg.dataset.n_train_steps,
			sampler_s_balanced=cfg.dataset.sampler_s_balanced,
			pre_computed_specs=cfg.dataset.pre_computed_specs,
		)
	elif cfg.dataset.acronym == 'CIFAR10':
		datamodule = CIFAR10DataModuleSSL(
			**datamodule_params,
			download_dataset=cfg.dataset.download,
		)
	elif cfg.dataset.acronym == 'ESC10':
		datamodule = ESC10DataModuleSSL(
			**datamodule_params,
			download_dataset=cfg.dataset.download,
			folds_train=cfg.dataset.folds_train,
			folds_val=cfg.dataset.folds_val,
		)
	elif cfg.dataset.acronym == 'FSD50K':
		datamodule = FSD50KDataModuleSSL(
			**datamodule_params,
			download_dataset=cfg.dataset.download,
			n_train_steps=cfg.dataset.n_train_steps,
			sampler_s_balanced=cfg.dataset.sampler_s_balanced,
		)
	elif cfg.dataset.acronym == 'GSC':
		datamodule = GSCDataModuleSSL(
			**datamodule_params,
			download_dataset=cfg.dataset.download,
		)
	elif cfg.dataset.acronym == 'PVC':
		datamodule = PVCDataModuleSSL(
			**datamodule_params,
			n_train_steps_u=cfg.dataset.n_train_steps,
		)
	elif cfg.dataset.acronym == 'UBS8K':
		datamodule = UBS8KDataModuleSSL(
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
