
from hydra.utils import DictConfig
from pytorch_lightning import LightningDataModule
from typing import Callable, Optional

from .ads import ADSSemiDataModule
from .cifar10 import CIFAR10SemiDataModule
from .esc10 import ESC10SemiDataModule
from .gsc import GSCSemiDataModule
from .pvc import PVCSemiDataModule
from .ubs8k import UBS8KSemiDataModule


def get_semi_datamodule_from_cfg(
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

	duplicate_loader_s = cfg.experiment.duplicate_loader_s if hasattr(cfg.experiment, "duplicate_loader_s") else False

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
		num_workers_s=round(cfg.cpus / 2),
		num_workers_u=round(cfg.cpus / 2),
		duplicate_loader_s=duplicate_loader_s,
	)

	if cfg.dataset.name == "ADS":
		datamodule = ADSSemiDataModule(
			**datamodule_params,
			train_subset=cfg.dataset.train_subset,
			nb_train_steps_u=cfg.dataset.nb_train_steps,
		)
	elif cfg.dataset.name == "CIFAR10":
		datamodule = CIFAR10SemiDataModule(
			**datamodule_params,
			download_dataset=cfg.dataset.download,
		)
	elif cfg.dataset.name == "ESC10":
		datamodule = ESC10SemiDataModule(
			**datamodule_params,
			download_dataset=cfg.dataset.download,
			folds_train=cfg.dataset.folds_train,
			folds_val=cfg.dataset.folds_val,
		)
	elif cfg.dataset.name == "GSC":
		datamodule = GSCSemiDataModule(
			**datamodule_params,
			download_dataset=cfg.dataset.download,
		)
	elif cfg.dataset.name == "PVC":
		datamodule = PVCSemiDataModule(
			**datamodule_params,
			nb_train_steps_u=cfg.dataset.nb_train_steps,
		)
	elif cfg.dataset.name == "UBS8K":
		datamodule = UBS8KSemiDataModule(
			**datamodule_params,
			folds_train=cfg.dataset.folds_train,
			folds_val=cfg.dataset.folds_val,
		)
	else:
		raise RuntimeError(
			f"Unknown dataset name '{cfg.dataset.name}'. "
			f"Must be one of {('ADS', 'CIFAR10', 'ESC10', 'GSC', 'PVC', 'UBS8K')}."
		)

	return datamodule
