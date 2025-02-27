import os

import hydra
import pyrootutils
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig, OmegaConf

from data_module import LichessDataModule
from gpt_module import GPT2Lightning
from ntfy import Ntfy


def main(config: DictConfig):

    dm = LichessDataModule(config)
    model = GPT2Lightning(config)
    trainer: Trainer = hydra.utils.instantiate(config.trainer)

    # with handle_ntfy_wandb(config, trainer) as (ntfy, run):
    if "lr_find" in config.run.keys() and config.run.lr_find:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model=model, datamodule=dm)
        ntfy = Ntfy(topic=os.environ.get("NTFY_TOPIC", None))
        ntfy.send_notification(f"{lr_finder.suggestion()=}")

    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":

    # torch.set_float32_matmul_precision("medium")  # take advantage of tensor cores

    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", ".env"],
        pythonpath=True,
        dotenv=True,
    )
    config_path = (
        os.path.join(root, os.environ.get("HYDRA_CONFIG_PATH", None))
        if os.environ.get("HYDRA_CONFIG_PATH", None)
        else str(root / "configs")
    )
    hydra.initialize(version_base=None, config_path=config_path, job_name="train-transformer")
    config = hydra.compose("train.yaml")
    print(OmegaConf.to_yaml(config))
    main(config)
