
import hydra
import os.path as osp
import torch

from hydra.utils import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from mlu.metrics import MetricDict, CategoricalAccuracy
from mlu.utils.misc import reset_seed

from sslh.callbacks import LogLRCallback
from sslh.datamodules.semi_supervised import get_semi_datamodule_from_cfg
from sslh.experiments.remixmatch import (
	ReMixMatch,
	ReMixMatchNoRot,
	ReMixMatchUnlabeledPreProcess,
)
from sslh.metrics import get_metrics
from sslh.models import get_model_from_cfg
from sslh.transforms import get_transform, get_target_transform, get_self_transform
from sslh.utils.get_from_name import (
	get_activation_from_name,
	get_criterion_from_name,
	get_optimizer_from_name,
	get_scheduler_from_name,
)
from sslh.utils.test_module import TestModule
from sslh.utils.test_stack_module import TestStackModule


@hydra.main(config_path="../config", config_name="remixmatch")
def main(cfg: DictConfig):
	# Initialisation
	reset_seed(cfg.seed)

	# Build transforms
	transform_weak = get_transform(cfg.dataset.name, cfg.experiment.augm_weak)
	transform_strong = get_transform(cfg.dataset.name, cfg.experiment.augm_strong)

	transform_train_s = transform_weak
	transform_train_u = ReMixMatchUnlabeledPreProcess(transform_weak, transform_strong, cfg.experiment.nb_augms)
	transform_val = get_transform(cfg.dataset.name, "identity")
	target_transform = get_target_transform(cfg.dataset.name)

	# Build datamodule
	datamodule = get_semi_datamodule_from_cfg(cfg, transform_train_s, transform_train_u, transform_val, target_transform)

	# Build model, activation, optimizer and criterions
	model = get_model_from_cfg(cfg)
	activation = get_activation_from_name(cfg.model.activation)
	optimizer = get_optimizer_from_name(cfg.optim.name, model, lr=cfg.optim.lr)

	criterion_s = get_criterion_from_name(cfg.experiment.criterion_s, cfg.experiment.reduction)
	criterion_u = get_criterion_from_name(cfg.experiment.criterion_u, cfg.experiment.reduction)
	criterion_u1 = get_criterion_from_name(cfg.experiment.criterion_u1, cfg.experiment.reduction)

	# Build metrics
	train_metrics, val_metrics, val_metrics_stack = get_metrics(cfg.dataset.name)
	metric_dict_train_s = MetricDict(**train_metrics, prefix="train/", suffix="_s")
	metric_dict_train_u_pseudo = MetricDict(**train_metrics, prefix="train/", suffix="_u")
	metric_dict_val = MetricDict(**val_metrics, prefix="val/")
	metric_dict_test = MetricDict(**val_metrics, prefix="test/")

	# Build Lightning module
	module_params = dict(
		model=model,
		optimizer=optimizer,
		activation=activation,
		criterion_s=criterion_s,
		criterion_u=criterion_u,
		criterion_u1=criterion_u1,
		metric_dict_train_s=metric_dict_train_s,
		metric_dict_train_u_pseudo=metric_dict_train_u_pseudo,
		metric_dict_val=metric_dict_val,
		metric_dict_test=metric_dict_test,
		log_on_epoch=cfg.dataset.log_on_epoch,
		lambda_u=cfg.experiment.lambda_u,
		lambda_u1=cfg.experiment.lambda_u1,
		nb_augms=cfg.experiment.nb_augms,
		temperature=cfg.experiment.temperature,
		alpha=cfg.experiment.alpha,
		history=cfg.experiment.history,
	)

	if cfg.experiment.fullname == "ReMixMatch":
		self_transform = get_self_transform(cfg.dataset.name)
		activation_r = get_activation_from_name(cfg.experiment.activation_r)
		criterion_r = get_criterion_from_name(cfg.experiment.criterion_r, cfg.experiment.reduction)
		metric_dict_train_r = MetricDict(acc=CategoricalAccuracy(), prefix="train/", suffix="_r")

		experiment_module = ReMixMatch(
			**module_params,
			self_transform=self_transform,
			activation_r=activation_r,
			criterion_r=criterion_r,
			lambda_r=cfg.experiment.lambda_r,
			metric_dict_train_r=metric_dict_train_r,
		)

	elif cfg.experiment.fullname == "ReMixMatch+NoRot":
		experiment_module = ReMixMatchNoRot(**module_params)

	else:
		raise RuntimeError(f"Unknown experiment name '{cfg.experiment.name}'.")

	# Prepare logger & callbacks
	callbacks = []
	logger = TensorBoardLogger(
		save_dir=osp.join(cfg.logdir, cfg.dataset.name),
		name=cfg.experiment.fullname,
		version=f"{cfg.datetime}{cfg.tag}",
	)

	checkpoint = ModelCheckpoint(
		dirpath=osp.join(logger.log_dir, "checkpoints"),
		save_last=True,
		save_top_k=1,
		verbose=True,
		monitor=cfg.dataset.monitor,
		mode=cfg.dataset.monitor_mode,
	)
	callbacks.append(checkpoint)

	if cfg.sched is not None:
		callbacks.append(LogLRCallback(log_on_epoch=cfg.sched.on_epoch))
		scheduler = get_scheduler_from_name(cfg.sched.fullname, optimizer, on_epoch=cfg.sched.on_epoch)
		callbacks.append(scheduler)

	# Start training
	trainer = Trainer(
		max_epochs=cfg.epochs,
		logger=logger,
		move_metrics_to_cpu=True,
		gpus=cfg.gpus,
		deterministic=True,
		max_steps=cfg.max_it,
		multiple_trainloader_mode="max_size_cycle",
		callbacks=callbacks,
		val_check_interval=cfg.dataset.val_check_interval,
	)
	trainer.fit(experiment_module, datamodule=datamodule)
	trainer.test(experiment_module, datamodule=datamodule)

	# Load best model before testing
	if osp.isfile(checkpoint.best_model_path):
		checkpoint_data = torch.load(checkpoint.best_model_path)
		experiment_module.load_state_dict(checkpoint_data['state_dict'])

	# Test
	trainer_params = dict(
		max_epochs=1,
		logger=logger,
		move_metrics_to_cpu=True,
		gpus=cfg.gpus,
		deterministic=True,
	)

	val_dataloader = datamodule.val_dataloader()
	test_dataloader = datamodule.test_dataloader()

	if val_dataloader is not None:
		if len(val_metrics) > 0:
			metric_dict_val = MetricDict(**val_metrics, prefix="val_best/")
			val_module = TestModule(experiment_module, metric_dict_val)
			trainer = Trainer(**trainer_params)
			trainer.test(val_module, val_dataloader)

		if len(val_metrics_stack) > 0:
			metric_dict_val_stack = MetricDict(**val_metrics_stack, prefix="val_stack_best/")
			val_stack_module = TestStackModule(experiment_module, metric_dict_val_stack)
			trainer = Trainer(**trainer_params)
			trainer.test(val_stack_module, val_dataloader)

	if test_dataloader is not None:
		if len(val_metrics) > 0:
			metric_dict_test = MetricDict(**val_metrics, prefix="test_best/")
			test_module = TestModule(experiment_module, metric_dict_test)
			trainer = Trainer(**trainer_params)
			trainer.test(test_module, test_dataloader)

		if len(val_metrics_stack) > 0:
			metric_dict_test_stack = MetricDict(**val_metrics_stack, prefix="test_stack_best/")
			test_stack_module = TestStackModule(experiment_module, metric_dict_test_stack)
			trainer = Trainer(**trainer_params)
			trainer.test(test_stack_module, test_dataloader)


if __name__ == "__main__":
	main()
