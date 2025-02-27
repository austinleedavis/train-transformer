import os
from contextlib import contextmanager

import hydra
import pyrootutils
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import Trainer
from omegaconf import DictConfig

import wandb
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

    with handle_ntfy_wandb(config):
        dm = LichessDataModule(config)
        model = GPT2Lightning(config)

        trainer: Trainer = hydra.utils.instantiate(config.trainer)

        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model)
        print(lr_finder.results)
        new_lr = lr_finder.suggestion()
        model.hparams.lr = new_lr

        trainer.fit(model=model, datamodule=dm)


@contextmanager
def handle_ntfy_wandb(config: DictConfig):
    ntfy = Ntfy(topic=os.environ.get("NTFY_TOPIC", None))
    try:
        hc = HydraConfig.get()
        project = hc.job.name
        name = "/".join(hc.run.dir.split("/")[1:])
        extra_headers = {"Title": f"{project} {name}"}
        if any(["WandbLogger" in list(cb.values())[0] for cb in config.trainer.logger]):
            run = wandb.init(project=project, name=name, config=OmegaConf.to_object(config))
            url = run.get_url()
            extra_headers["Click"] = url

        ntfy.send_notification(f"🤖Started", extra_headers=extra_headers)
        final_text = f"🏆️ Finished"
        exit_code = 0

        yield
    except Exception as e:
        final_text = f"💢 Exception occurred {e}"
        exit_code = -1
        raise
    finally:
        ntfy.send_notification(final_text, extra_headers=extra_headers)
        if run:
            run.finish(exit_code)


if __name__ == "__main__":
    main()
