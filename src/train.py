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

_root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", ".env"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": (
        os.path.join(_root, os.environ.get("HYDRA_CONFIG_PATH", None))
        if os.environ.get("HYDRA_CONFIG_PATH", None)
        else str(_root / "configs")
    ),
    "config_name": "train.yaml",
}


@hydra.main(**_HYDRA_PARAMS)
def main(config: DictConfig):

    torch.set_float32_matmul_precision("medium")  # take advantage of tensor cores

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

    main()
