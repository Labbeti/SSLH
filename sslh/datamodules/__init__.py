
from hydra.utils import DictConfig
from pytorch_lightning import LightningDataModule

from sslh.datamodules.fully_supervised import (
	ADSFullyDataModule,
	CIFAR10FullyDataModule,
	ESC10FullyDataModule,
	GSCFullyDataModule,
	UBS8KFullyDataModule,
)

from sslh.datamodules.partial_supervised import (
	ADSPartialDataModule,
	CIFAR10PartialDataModule,
	ESC10PartialDataModule,
	GSCPartialDataModule,
	UBS8KPartialDataModule,
)

from sslh.datamodules.semi_supervised import (
	ADSSemiDataModule,
	CIFAR10SemiDataModule,
	ESC10SemiDataModule,
	GSCSemiDataModule,
	PVCSemiDataModule,
	UBS8KSemiDataModule,
)


def get_datamodule_from_cfg(cfg: DictConfig) -> LightningDataModule:
	train_type = cfg.train_type
	dataset_name = cfg.dataset.name

	datasets_available = ('ads', 'cifar10', 'esc10', 'gsc', 'ubs8k')

	if train_type == "fully_supervised":
		if dataset_name == "ads":
			return ADSFullyDataModule(**cfg.dataset)
		elif dataset_name == "cifar10":
			return CIFAR10FullyDataModule(**cfg.dataset)
		elif dataset_name == "esc10":
			return ESC10FullyDataModule(**cfg.dataset)
		elif dataset_name == "gsc":
			return GSCFullyDataModule(**cfg.dataset)
		elif dataset_name == "ubs8k":
			return UBS8KFullyDataModule(**cfg.dataset)
		else:
			raise RuntimeError(
				f"Unknown dataset name '{dataset_name}' for train '{train_type}'. Datasets available are : {datasets_available}.")
	else:
		raise RuntimeError(f"Unknown train '{train_type}'.")
