
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
