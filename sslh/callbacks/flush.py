
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer
from torch.utils.tensorboard.writer import SummaryWriter


class FlushLoggerCallback(Callback):
	def __init__(self):
		super().__init__()

	def _flush_expt(self, pl_module: LightningModule):
		experiment = pl_module.logger.experiment
		if isinstance(experiment, SummaryWriter):
			experiment.flush()
		else:
			raise RuntimeError(f'Unknown experiment type "{type(experiment)}" for FlushLoggerCallback.')

	def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
		self._flush_expt(pl_module)
