#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import os.path as osp

import hydra
import yaml

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from sslh.utils.collections import flat_dict_of_dict


@hydra.main(
    config_path=osp.join("..", "conf"),
    config_name="supervised",
)
def run_print_cfg(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    flat_cfg = flat_dict_of_dict(cfg_dict)  # type: ignore
    # note : use default_flow_style=None to collapse list to one line
    print(yaml.dump(flat_cfg, sort_keys=False, default_flow_style=None), end="")

    hydra_cfg = HydraConfig.get()
    choices = dict(hydra_cfg.runtime.choices)
    print(yaml.dump(choices, sort_keys=False))


if __name__ == "__main__":
    run_print_cfg()
