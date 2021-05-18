"""
	Pseudo-Labeling (PL) training.
"""
import hydra
import logging
import os.path as osp
import torch

from hydra.utils import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from mlu.utils.misc import reset_seed

from sslh.callbacks import LogLRCallback, FlushLoggerCallback, LogAttributeCallback, WarmUpCallback
from sslh.datamodules.semi_supervised.get_from_cfg import get_datamodule_ssl_from_cfg
from sslh.expt.pseudo_labeling import (
	PseudoLabeling,
)
from sslh.metrics.get_from_name import get_metrics
from sslh.models.get_from_name import get_model_from_name
from sslh.transforms.get_from_name import get_transform, get_target_transform
from sslh.utils.custom_logger import CustomTensorboardLogger
from sslh.utils.get_obj_from_name import (
	get_activation_from_name,
	get_criterion_from_name,
	get_optimizer_from_name,
	get_scheduler_from_name,
)
from sslh.utils.test_module import TestModule
from sslh.utils.test_stack_module import TestStackModule


@hydra.main(config_path='../config', config_name='pseudo_labeling')
def main(cfg: DictConfig):
	# Initialisation
	reset_seed(cfg.seed)
	if cfg.verbose:
		logging.info(f'Configuration:\n{OmegaConf.to_yaml(cfg):s}')
		logging.info(f'Datetime: {cfg.datetime:s}\n')
	torch.autograd.set_detect_anomaly(cfg.debug)

	# Build transforms
	transform_identity = get_transform(cfg.dataset.acronym, 'identity', **cfg.dataset.transform)
	transform_train_s = transform_identity
	transform_train_u = transform_identity
	transform_val = transform_identity
	target_transform = get_target_transform(cfg.dataset.acronym)

	# Build datamodule
	datamodule = get_datamodule_ssl_from_cfg(cfg, transform_train_s, transform_train_u, transform_val, target_transform)

	# Build model, activation, optimizer and criterion
	model = get_model_from_name(**cfg.model)
	activation = get_activation_from_name(cfg.expt.activation)
	optimizer = get_optimizer_from_name(**cfg.optim, parameters=model)
	criterion_s = get_criterion_from_name(cfg.expt.criterion_s, cfg.expt.reduction)
	criterion_u = get_criterion_from_name(cfg.expt.criterion_u, cfg.expt.reduction)

	# Build metrics
	train_metrics, val_metrics, val_metrics_stack = get_metrics(cfg.dataset.acronym)

	# Build Lightning module
	module_params = dict(
		model=model,
		optimizer=optimizer,
		activation=activation,
		criterion_s=criterion_s,
		criterion_u=criterion_u,
		target_transform=target_transform,
		lambda_u=cfg.expt.lambda_u,
		threshold=cfg.expt.threshold,
		train_metrics=train_metrics,
		val_metrics=val_metrics,
		log_on_epoch=cfg.dataset.log_on_epoch,
	)

	if cfg.expt.name == 'PseudoLabeling':
		pl_module = PseudoLabeling(
			**module_params,
		)

	else:
		raise RuntimeError(f'Unknown experiment name "{cfg.expt.name}".')

	# Prepare logger & callbacks
	logger = CustomTensorboardLogger(**cfg.logger, additional_params=cfg)

	callbacks = []
	checkpoint = ModelCheckpoint(osp.join(logger.log_dir, 'checkpoints'), **cfg.checkpoint)
	callbacks.append(checkpoint)
	flush_callback = FlushLoggerCallback()
	callbacks.append(flush_callback)

	if cfg.sched.name != 'none':
		callbacks.append(LogLRCallback(log_on_epoch=cfg.sched.on_epoch))
		scheduler = get_scheduler_from_name(cfg.sched.name, optimizer, on_epoch=cfg.sched.on_epoch)
		callbacks.append(scheduler)

	log_gpu_memory = 'all' if cfg.debug else None

	if cfg.warmup.name == 'linear':
		warmup = WarmUpCallback(
			target_value=cfg.expt.lambda_u,
			target_obj=pl_module,
			target_attribute='lambda_u',
			n_steps=cfg.warmup.n_steps,
			ratio_n_steps=cfg.warmup.ratio_n_steps,
			on_epoch=cfg.warmup.on_epoch,
		)
		callbacks.append(warmup)
	callbacks.append(LogAttributeCallback('lambda_u', log_on_epoch=cfg.warmup.on_epoch))

	# Resume model weights with checkpoint
	if cfg.resume_path is not None:
		if not isinstance(cfg.resume_path, str) or not osp.isfile(cfg.resume_path):
			raise RuntimeError(f'Invalid resume checkpoint filepath "{cfg.resume_path}".')
		checkpoint_data = torch.load(cfg.resume_path)
		pl_module.load_state_dict(checkpoint_data['state_dict'])

	# Start training
	trainer = Trainer(
		**cfg.trainer,
		logger=logger,
		callbacks=callbacks,
		log_gpu_memory=log_gpu_memory,
	)

	trainer.fit(pl_module, datamodule=datamodule)
	trainer.test(pl_module, datamodule=datamodule)

	# Load best model before testing
	if osp.isfile(checkpoint.best_model_path):
		checkpoint_data = torch.load(checkpoint.best_model_path)
		pl_module.load_state_dict(checkpoint_data['state_dict'])

	# Test with validation/testing and non-stack or stack metrics.
	val_dataloader = datamodule.val_dataloader()
	test_dataloader = datamodule.test_dataloader()

	val_or_test_modules = [
		TestModule(pl_module, val_metrics, 'val_best/'),
		TestStackModule(pl_module, val_metrics_stack, 'val_stack_best/'),
		TestModule(pl_module, val_metrics, 'test_best/'),
		TestStackModule(pl_module, val_metrics_stack, 'test_stack_best/'),
	]
	val_or_test_dataloaders = [
		val_dataloader,
		val_dataloader,
		test_dataloader,
		test_dataloader,
	]

	for module, dataloader in zip(val_or_test_modules, val_or_test_dataloaders):
		if len(module.metric_dict) > 0 and dataloader is not None:
			trainer.test_dataloaders = None
			trainer.test(module, dataloader)

	logger.save_and_close()


if __name__ == '__main__':
	main()
