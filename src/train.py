import os

import hydra
import pyrootutils
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from data_module import LichessDataModule
from gpt_module import GPT2Lightning
from ntfy import Ntfy

torch.set_float32_matmul_precision("medium")  # take advantage of tensor cores

root = pyrootutils.setup_root(
    search_from=__file__, indicator=[".git", ".env"], pythonpath=True, dotenv=True
)

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": (
        os.path.join(root, os.environ.get("HYDRA_CONFIG_PATH", None))
        if os.environ.get("HYDRA_CONFIG_PATH", None)
        else str(root / "configs")
    ),
    "config_name": "train.yaml",
}


@hydra.main(**_HYDRA_PARAMS)
def main(config: DictConfig):
    print(config)
    with Ntfy(topic=os.environ.get("NTFY_TOPIC", None)).context("Job"):

        dm = LichessDataModule(config)
        model = GPT2Lightning(config)

        trainer: Trainer = hydra.utils.instantiate(config.trainer)

        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model)
        print(lr_finder.results)
        new_lr = lr_finder.suggestion()
        model.hparams.lr = new_lr

        trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
