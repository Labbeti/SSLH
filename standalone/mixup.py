
import hydra
import os.path as osp

from hydra.utils import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from mlu.metrics import MetricDict
from mlu.utils.misc import reset_seed

from sslh.callbacks import LogLRCallback
from sslh.datamodules.fully_supervised import get_fully_datamodule_from_cfg
from sslh.datamodules.partial_supervised import get_partial_datamodule_from_cfg
from sslh.experiments.mixup import MixUp
from sslh.metrics import get_metrics
from sslh.models import get_model_from_cfg
from sslh.transforms import get_transform, get_target_transform
from sslh.utils.get_from_name import (
	get_activation_from_name,
	get_criterion_from_name,
	get_optimizer_from_name,
	get_scheduler_from_name,
)
from sslh.utils.test_stack_module import TestStackModule


@hydra.main(config_path="../config", config_name="mixup")
def main(cfg: DictConfig):
	# Initialisation
	reset_seed(cfg.seed)

	# Build transforms
	transform_train = get_transform(cfg.dataset.name, cfg.experiment.augm_train)
	transform_val = get_transform(cfg.dataset.name, "identity")
	target_transform = get_target_transform(cfg.dataset.name)

	# Build datamodule
	if cfg.ratio >= 1.0:
		datamodule = get_fully_datamodule_from_cfg(cfg, transform_train, transform_val, target_transform)
	else:
		datamodule = get_partial_datamodule_from_cfg(cfg, transform_train, transform_val, target_transform)

	# Build model
	model = get_model_from_cfg(cfg)
	activation = get_activation_from_name(cfg.model.activation)
	optimizer = get_optimizer_from_name(cfg.optim.name, model, lr=cfg.optim.lr)
	criterion = get_criterion_from_name(cfg.experiment.criterion, cfg.experiment.reduction)

	# Build metrics
	train_metrics, val_metrics, val_metrics_stack = get_metrics(cfg.dataset.name)
	metric_dict_train = MetricDict(**train_metrics, prefix="train/")
	metric_dict_val = MetricDict(**val_metrics, prefix="val/")
	metric_dict_test = MetricDict(**val_metrics, prefix="test/")

	# Build Lightning module
	module_params = dict(
		model=model,
		optimizer=optimizer,
		activation=activation,
		criterion=criterion,
		metric_dict_train=metric_dict_train,
		metric_dict_val=metric_dict_val,
		metric_dict_test=metric_dict_test,
		log_on_epoch=cfg.dataset.log_on_epoch,
	)

	if cfg.experiment.fullname == "MixUp":
		experiment_module = MixUp(**module_params, alpha=cfg.experiment.alpha)
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
		max_steps=cfg.max_it,
		multiple_trainloader_mode="max_size_cycle",
		logger=logger,
		move_metrics_to_cpu=True,
		gpus=cfg.gpus,
		callbacks=callbacks,
		val_check_interval=cfg.dataset.val_check_interval,
		deterministic=True,
	)
	trainer.fit(experiment_module, datamodule=datamodule)

	# Test
	trainer.test(experiment_module, datamodule=datamodule)

	# If some metrics need to be computed with all predictions, test with "TestStackModule"
	if len(val_metrics_stack) > 0:
		metric_dict_val_stack = MetricDict(**val_metrics_stack, prefix="val_stack/")
		metric_dict_test_stack = MetricDict(**val_metrics_stack, prefix="test_stack/")

		val_dataloader = datamodule.val_dataloader()
		test_dataloader = datamodule.test_dataloader()

		trainer_params = dict(
			max_epochs=1,
			logger=logger,
			move_metrics_to_cpu=True,
			gpus=cfg.gpus,
		)
		trainer = Trainer(**trainer_params)
		stack_module = TestStackModule(experiment_module, metric_dict_val_stack)
		trainer.test(stack_module, test_dataloaders=val_dataloader)

		trainer = Trainer(**trainer_params)
		stack_module = TestStackModule(experiment_module, metric_dict_test_stack)
		trainer.test(stack_module, test_dataloaders=test_dataloader)


if __name__ == "__main__":
	main()
