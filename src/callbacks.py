import os

from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger

# from lightning.pytorch.loggers.wandb
from src.ntfy import Ntfy


class NtfyCallback(Callback):

    def __init__(self):
        super().__init__()
        self.ntfy = Ntfy(topic=os.environ.get("NTFY_TOPIC", None))

    def setup(self, trainer, pl_module, stage):
        if trainer.global_rank == 0:
            hc = HydraConfig.get()
            project = pl_module.config.run.project
            name = "/".join(hc.run.dir.split("/")[1:])
            extra_headers = {"Title": f"{project} {name}"}

            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    run = logger.experiment
                    url = run.get_url()
                    extra_headers["Click"] = url
                    break

            self.ntfy.send_notification(f"🤖 {stage} started", extra_headers=extra_headers)

    def teardown(self, trainer, pl_module, stage):
        if trainer.global_rank == 0:
            hc = HydraConfig.get()
            project = pl_module.config.run.project
            name = "/".join(hc.run.dir.split("/")[1:])
            extra_headers = {"Title": f"{project} {name}"}

            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    run = logger.experiment
                    url = run.get_url()
                    extra_headers["Click"] = url
                    break

            self.ntfy.send_notification(f"🏆️ {stage} finished", extra_headers=extra_headers)

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0:
            hc = HydraConfig.get()
            project = pl_module.config.run.project
            name = "/".join(hc.run.dir.split("/")[1:])
            extra_headers = {"Title": f"{project} {name}"}

            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    run = logger.experiment
                    url = run.get_url()
                    extra_headers["Click"] = url
                    break

            e = "Keyboard interrupt" if isinstance(exception, KeyboardInterrupt) else str(exception)

            self.ntfy.send_notification(f"💢 Exception: {e}", extra_headers=extra_headers)


# @contextmanager
# def handle_ntfy_wandb(config: DictConfig):
#
#     try:
#         hc = HydraConfig.get()
#         project = config.run.project
#         name = "/".join(hc.run.dir.split("/")[1:])
#         extra_headers = {"Title": f"{project} {name}"}
#         if any(["WandbLogger" in list(cb.values())[0] for cb in config.trainer.logger]):
#             run = wandb.init(project=project, name=name, config=OmegaConf.to_object(config))
#             url = run.get_url()
#             extra_headers["Click"] = url

#         ntfy.send_notification(f"🤖Started", extra_headers=extra_headers)
#         final_text = f"🏆️ Finished"
#         exit_code = 0

#         yield (ntfy, run)
#     except Exception as e:
#         final_text = f"💢 Exception occurred {e}"
#         exit_code = -1
#         raise
#     finally:
#         ntfy.send_notification(final_text, extra_headers=extra_headers)
#         if run:
#             run.finish(exit_code)
