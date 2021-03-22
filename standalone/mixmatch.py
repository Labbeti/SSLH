
import hydra
import os.path as osp

from hydra.utils import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from mlu.metrics import MetricDict
from mlu.utils.misc import reset_seed

from sslh.callbacks import LogLRCallback, LogAttributeCallback, WarmUpCallback
from sslh.datamodules.semi_supervised import (
	ADSSemiDataModule,
	CIFAR10SemiDataModule,
	ESC10SemiDataModule,
	GSCSemiDataModule,
	PVCSemiDataModule,
	UBS8KSemiDataModule,
)
from sslh.experiments import MixMatch, MixMatchNoMixUp, MixMatchUnlabeledPreProcess
from sslh.metrics import get_metrics
from sslh.models import get_model_from_cfg
from sslh.transforms import get_transform, get_target_transform
from sslh.utils.get_from_name import (
	get_criterion_from_name, get_activation_from_name, get_optimizer_from_name, get_scheduler_from_name
)
from sslh.utils.test_stack_module import TestStackModule


@hydra.main(config_path="../config", config_name="mixmatch")
def main(cfg: DictConfig):
	# Initialisation
	reset_seed(cfg.seed)

	# Build transforms
	transform_weak = get_transform(cfg.dataset.name, cfg.experiment.augm_weak)

	transform_train_s = transform_weak
	transform_train_u = MixMatchUnlabeledPreProcess(transform_weak, cfg.experiment.nb_augms)
	transform_val = get_transform(cfg.dataset.name, "identity")
	target_transform = get_target_transform(cfg.dataset.name)

	# Build datamodule
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

	# Build model, activation, optimizer and criterions
	model = get_model_from_cfg(cfg)
	activation = get_activation_from_name(cfg.model.activation)
	optimizer = get_optimizer_from_name(cfg.optim.name, model, lr=cfg.optim.lr)
	criterion_s = get_criterion_from_name(cfg.experiment.criterion_s, cfg.experiment.reduction)
	criterion_u = get_criterion_from_name(cfg.experiment.criterion_u, cfg.experiment.reduction)

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
		metric_dict_train_s=metric_dict_train_s,
		metric_dict_train_u_pseudo=metric_dict_train_u_pseudo,
		metric_dict_val=metric_dict_val,
		metric_dict_test=metric_dict_test,
		log_on_epoch=cfg.dataset.log_on_epoch,
		lambda_u=cfg.experiment.lambda_u,
		nb_augms=cfg.experiment.nb_augms,
		temperature=cfg.experiment.temperature,
	)

	if cfg.experiment.fullname == "MixMatch":
		experiment_module = MixMatch(
			**module_params,
			alpha=cfg.experiment.alpha,
		)

	elif cfg.experiment.fullname == "MixMatch+NoMixUp":
		experiment_module = MixMatchNoMixUp(
			**module_params,
		)

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

	warmup = WarmUpCallback(
		target_value=cfg.experiment.lambda_u,
		target_num_steps=cfg.experiment.warmup_num_steps,
		target_ratio_num_steps=cfg.experiment.warmup_ratio_num_steps,
		target_obj=experiment_module,
		target_attribute="lambda_u",
		on_train_batch_end=True,
		on_train_epoch_end=False,
	)
	callbacks.append(warmup)
	callbacks.append(LogAttributeCallback("lambda_u"))

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
