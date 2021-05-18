
import logging
import os.path as osp
import time

from argparse import Namespace
from omegaconf import Container, DictConfig, OmegaConf
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from typing import Any, Dict, Optional, Union

from mlu.utils.misc import get_current_git_hash


class CustomTensorboardLogger(TensorBoardLogger):
	NAME_METRICS_FILE = 'metrics.yaml'

	def __init__(
		self,
		save_dir: str,
		name: Optional[str] = 'default',
		version: Optional[Union[int, str]] = None,
		log_graph: bool = False,
		default_hp_metric: bool = True,
		prefix: str = '',
		additional_params: Union[Dict[str, Any], DictConfig, None] = None,
		**kwargs,
	):
		if additional_params is None:
			additional_params = {}

		super().__init__(
			save_dir=save_dir,
			name=name,
			version=version,
			log_graph=log_graph,
			default_hp_metric=default_hp_metric,
			prefix=prefix,
			**kwargs,
		)
		self.params_merged = {}
		self.metrics_merged = {}

		self._start = time.time()
		self._custom_expt = None
		self._closed = False

		self.params_merged.update(additional_params)
		git_hash = get_current_git_hash()
		self.params_merged['git_hash'] = git_hash

		if default_hp_metric:
			self.metrics_merged['hp_metric'] = -1

		if isinstance(additional_params, Container):
			params = str(OmegaConf.to_container(additional_params, True))
		else:
			params = str(additional_params)
		self.experiment.add_text(f'{self.version}/params', params)
		self.experiment.add_text(f'{self.version}/git_hash', git_hash)

	@staticmethod
	def from_cfg(cfg: DictConfig) -> 'CustomTensorboardLogger':
		return CustomTensorboardLogger(**cfg.logger, additional_params=cfg)

	def log_hyperparams(
		self,
		params: Union[Dict[str, Any], Namespace],
		metrics: Optional[Dict[str, Any]] = None,
	):
		if metrics is None:
			metrics = {}
		self.params_merged.update(params)
		self.metrics_merged.update(metrics)
		self.experiment.flush()

	def finalize(self, status: str):
		self.experiment.flush()
		self.save()

	def save_and_close(self):
		if self._closed:
			raise RuntimeError('CustomTensorboardLogger cannot be closed twice.')

		duration = self.get_duration()
		self.params_merged['duration'] = duration
		self.experiment.add_text(f'{self.version}/duration', duration)

		self.params_merged = dict(sorted(self.params_merged.items()))
		self.metrics_merged = dict(sorted(self.metrics_merged.items()))

		def convert_val(v):
			return v.item() if isinstance(v, Tensor) else v
		self.metrics_merged = {k: convert_val(v) for k, v in self.metrics_merged.items()}

		fpath_metrics = osp.join(self.log_dir, self.NAME_METRICS_FILE)
		if osp.isdir(self.log_dir) and not osp.isfile(fpath_metrics):
			save_hparams_to_yaml(fpath_metrics,  self.metrics_merged)

		logging.info(f'Saving metrics : \n{OmegaConf.to_yaml(self.metrics_merged)}')
		super().log_hyperparams(self.params_merged, self.metrics_merged)
		self.experiment.flush()

		self._closed = True

	@property
	def hparams(self) -> dict:
		return self.params_merged

	@hparams.setter
	def hparams(self, other: dict):
		self.params_merged = other

	def get_duration(self) -> str:
		duration = int(time.time() - self._start)
		rest, seconds = divmod(duration, 60)
		hours, minutes = divmod(rest, 60)
		duration_str = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
		return duration_str
