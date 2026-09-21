from typing import Any

import wandb
from omegaconf import OmegaConf

from utils.easylogger import LoggerHandler


class WandbLoggerHandler(LoggerHandler):
    def __init__(self, cfg):
        if cfg.agent.name == 'FOIL':
            wandb.init(
                entity='chr0nix',
                project='FOIL ICML',
                name=f'{cfg.project.name}_{cfg.agent.name}_{cfg.env.name}_{cfg.seed}',
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        else:
            wandb.init(
                entity='chr0nix',
                project='FOIL Baselines Final',
                name=f'{cfg.project.name}_{cfg.agent.name}_{cfg.env.name}_{cfg.seed}',
                config=OmegaConf.to_container(cfg, resolve=True),
            )

    def writekvs(self, kvs: dict):
        wandb.log(kvs, step=kvs['step'])