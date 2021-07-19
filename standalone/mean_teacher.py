"""
	Mean Teacher (MT) training.
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
from sslh.expt.mean_teacher import (
	MeanTeacher,
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


@hydra.main(config_path=osp.join('..', 'config'), config_name='mean_teacher')
def main(cfg: DictConfig) -> None:
	# Initialisation
	reset_seed(cfg.seed)
	if cfg.verbose:
		logging.info(f'Configuration:\n{OmegaConf.to_yaml(cfg):s}')
		logging.info(f'Datetime: {cfg.datetime:s}\n')
	torch.autograd.set_detect_anomaly(cfg.debug)

	# Build transforms
	transform_train_s = get_transform(cfg.data.acronym, 'identity', **cfg.data.transform)
	transform_train_u = get_transform(cfg.data.acronym, 'identity', **cfg.data.transform)
	transform_val = get_transform(cfg.data.acronym, 'identity', **cfg.data.transform)
	target_transform = get_target_transform(cfg.data.acronym)

	# Build datamodule
	datamodule = get_datamodule_ssl_from_cfg(cfg, transform_train_s, transform_train_u, transform_val, target_transform)

	# Build model, activation, optimizer and criterion
	student = get_model_from_name(**cfg.model)
	teacher = get_model_from_name(**cfg.model)
	activation = get_activation_from_name(cfg.expt.activation)
	optimizer = get_optimizer_from_name(**cfg.optim, parameters=student)

	criterion_s = get_criterion_from_name(cfg.expt.criterion_s, cfg.expt.reduction)
	criterion_ccost = get_criterion_from_name(cfg.expt.criterion_ccost, cfg.expt.reduction)

	# Build metrics
	train_metrics, val_metrics, val_metrics_stack = get_metrics(cfg.data.acronym)

	# Build Lightning module
	module_params = dict(
		student=student,
		teacher=teacher,
		optimizer=optimizer,
		activation=activation,
		criterion_s=criterion_s,
		criterion_ccost=criterion_ccost,
		decay=cfg.expt.decay,
		lambda_ccost=cfg.expt.lambda_ccost,
		train_metrics=train_metrics,
		val_metrics=val_metrics,
		log_on_epoch=cfg.data.log_on_epoch,
	)

	if cfg.expt.name == 'MeanTeacher':
		pl_module = MeanTeacher(**module_params)

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
			target_attribute='lambda_ccost',
			n_steps=cfg.warmup.n_steps,
			ratio_n_steps=cfg.warmup.ratio_n_steps,
			on_epoch=cfg.warmup.on_epoch,
		)
		callbacks.append(warmup)
	callbacks.append(LogAttributeCallback('lambda_ccost', log_on_epoch=cfg.warmup.on_epoch))

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
			trainer.test_dataloaders = []
			trainer.test(module, dataloader)

	logger.save_and_close()


if __name__ == '__main__':
	main()
