#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MixMatch (MM) training. """

import logging
import os.path as osp
import warnings

from typing import Any

import hydra
import torch

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from sslh.callbacks.flush import FlushLoggerCallback
from sslh.callbacks.log import LogLRCallback, LogPLAttrCallback
from sslh.callbacks.time import TimeTrackerCallback
from sslh.pl_modules.mixmatch import (
    MixMatch,
    MixMatchMultiSharp,
    MixMatchNoMixUp,
    MixMatchUnlabeledPreProcess,
)
from sslh.metrics.get.get_from_name import get_metrics
from sslh.transforms.get.get_from_name import get_transform, get_target_transform
from sslh.utils.custom_logger import CustomTensorboardLogger
from sslh.utils.get_obj_from_name import (
    get_activation_from_name,
    get_criterion_from_name,
)
from sslh.utils.misc import reset_seed
from sslh.utils.test_module import TestModule
from sslh.utils.test_stack_module import TestStackModule


pylog = logging.getLogger(__name__)


@hydra.main(config_path=osp.join("..", "conf"), config_name="mixmatch")
def main_mixmatch(cfg: DictConfig) -> None:
    # Initialisation
    reset_seed(cfg.seed)
    if cfg.verbose <= 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif cfg.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if cfg.verbose:
        pylog.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        pylog.info(f"Datetime: {cfg.datetime}\n")
    torch.autograd.set_detect_anomaly(bool(cfg.debug))  # type: ignore

    # Ignore torch.fft user warnings
    if cfg.ignore_user_warnings:
        warnings.simplefilter("ignore", UserWarning, append=True)

    hydra_cfg = HydraConfig.get()
    data_name = hydra_cfg.runtime.choices["data"]
    pl_name = hydra_cfg.runtime.choices["pl"]

    if not data_name.startswith("ssl"):
        raise ValueError(
            f"Invalid data={data_name}. (expected an semi-supervised datamodule)"
        )

    # Build transforms
    weak_transform = get_transform(data_name, cfg.weak_aug, **cfg.data.spectro)

    train_transform_s = weak_transform
    train_transform_u = MixMatchUnlabeledPreProcess(weak_transform, cfg.pl.n_augms)
    val_transform = get_transform(data_name, cfg.val_aug, **cfg.data.spectro)
    target_transform = get_target_transform(data_name)

    # Build datamodule
    datamodule = hydra.utils.instantiate(
        cfg.data.dm,
        train_transform_s=train_transform_s,
        train_transform_u=train_transform_u,
        val_transform=val_transform,
        target_transform=target_transform,
    )

    # Build model, activation, optimizer and criterion
    model = hydra.utils.instantiate(cfg.model)
    activation = get_activation_from_name(cfg.pl.activation)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    criterion_s = get_criterion_from_name(cfg.pl.criterion_s, cfg.pl.reduction)
    criterion_u = get_criterion_from_name(cfg.pl.criterion_u, cfg.pl.reduction)

    # Build metrics
    train_metrics, val_metrics, val_metrics_stack = get_metrics(data_name)

    # Build Lightning module
    module_params: dict[str, Any] = dict(
        model=model,
        optimizer=optimizer,
        activation=activation,
        criterion_s=criterion_s,
        criterion_u=criterion_u,
        lambda_u=cfg.pl.lambda_u,
        n_augms=cfg.pl.n_augms,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        log_on_epoch=cfg.data.log_on_epoch,
    )

    if pl_name == "mixmatch":
        pl_module = MixMatch(
            **module_params,
            temperature=cfg.pl.temperature,
            alpha=cfg.pl.alpha,
        )

    elif pl_name == "mixmatch_multi_sharp":
        pl_module = MixMatchMultiSharp(
            **module_params,
            alpha=cfg.pl.alpha,
            sharpen_threshold=cfg.pl.sharpen_threshold,
        )

    elif pl_name == "mixmatch_nomixup":
        pl_module = MixMatchNoMixUp(
            **module_params,
            temperature=cfg.pl.temperature,
        )

    else:
        raise RuntimeError(f'Unknown experiment name "{pl_name}".')

    # Prepare logger & callbacks
    logger = CustomTensorboardLogger(**cfg.logger, additional_params=cfg)

    callbacks = []
    checkpoint = ModelCheckpoint(osp.join(logger.log_dir, "checkpoints"), **cfg.ckpt)
    callbacks.append(checkpoint)

    flush_callback = FlushLoggerCallback()
    callbacks.append(flush_callback)

    scheduler = hydra.utils.instantiate(cfg.sched, optimizer=optimizer)
    if scheduler is not None:
        callbacks.append(scheduler)
        sched_on_epoch = cfg.sched.on_epoch
    else:
        sched_on_epoch = True

    log_lr = LogLRCallback(log_on_epoch=sched_on_epoch)
    callbacks.append(log_lr)

    time_tracker = TimeTrackerCallback()
    callbacks.append(time_tracker)

    log_gpu_memory = "all" if cfg.debug else None

    warmup = hydra.utils.instantiate(cfg.warmup, target_obj=pl_module)
    if warmup is not None:
        callbacks.append(warmup)

    if hasattr(cfg.warmup, "target_attr") and hasattr(cfg.warmup, "on_epoch"):
        log_attr = LogPLAttrCallback(
            cfg.warmup.target_attr, log_on_epoch=cfg.warmup.on_epoch
        )
        callbacks.append(log_attr)

    # Resume model weights with checkpoint
    if cfg.resume_path is not None:
        if not isinstance(cfg.resume_path, str) or not osp.isfile(cfg.resume_path):
            raise RuntimeError(
                f'Invalid resume checkpoint filepath "{cfg.resume_path}".'
            )
        checkpoint_data = torch.load(cfg.resume_path, map_location=pl_module.device)
        pl_module.load_state_dict(checkpoint_data["state_dict"])

    # Start training
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        log_gpu_memory=log_gpu_memory,
    )

    trainer.fit(pl_module, datamodule=datamodule)
    trainer.test(pl_module, datamodule=datamodule)

    # Load best model before testing
    if osp.isfile(checkpoint.best_model_path):
        checkpoint_data = torch.load(checkpoint.best_model_path)
        pl_module.load_state_dict(checkpoint_data["state_dict"])

    # Test with validation/testing and non-stack or stack metrics.
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    val_or_test_modules = [
        TestModule(pl_module, val_metrics, "val_best/"),
        TestStackModule(pl_module, val_metrics_stack, "val_stack_best/"),
        TestModule(pl_module, val_metrics, "test_best/"),
        TestStackModule(pl_module, val_metrics_stack, "test_stack_best/"),
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

    metrics = {
        "fit_duration_h": time_tracker.get_fit_duration_in_hours(),
        "test_duration_h": time_tracker.get_test_duration_in_hours(),
    }
    logger.log_hyperparams(metrics=metrics)
    logger.save_and_close()

    if cfg.verbose >= 1:
        pylog.info(f"Results are saved in directory '{logger.log_dir}'.")


if __name__ == "__main__":
    main_mixmatch()
