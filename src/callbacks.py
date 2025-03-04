import os
import threading

from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger

from src.ntfy import Ntfy


class NtfyCallback(Callback):

    _stop_training: bool = False
    run_name: str = "00-00-00"
    """Keyword to stop training."""

    def __init__(self):
        super().__init__()
        self.ntfy = Ntfy(topic=os.environ.get("NTFY_TOPIC", None))
        threading.Thread(
            target=self.ntfy.subscribe, args=(self.handle_message,), daemon=True
        ).start()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._stop_training:
            trainer.should_stop = True

    def handle_message(self, message):
        if message.strip().lower().strip() == self.run_name:
            self._stop_training = True

    def setup(self, trainer, pl_module, stage):
        if trainer.global_rank == 0:
            extra_headers = self.get_extra_headers(trainer, pl_module)
            self.ntfy.send_notification(f"🤖 {stage} started", extra_headers=extra_headers)

    def teardown(self, trainer, pl_module, stage):
        if trainer.global_rank == 0:
            extra_headers = self.get_extra_headers(trainer, pl_module)
            self.ntfy.send_notification(f"🏆️ {stage} finished", extra_headers=extra_headers)

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0:
            extra_headers = self.get_extra_headers(trainer, pl_module)
            e = "Keyboard interrupt" if isinstance(exception, KeyboardInterrupt) else str(exception)
            self.ntfy.send_notification(f"💢 Exception: {e}", extra_headers=extra_headers)

    def get_extra_headers(self, trainer, pl_module):
        hc = HydraConfig.get()
        project = pl_module.config.run.project
        name = "/".join(hc.run.dir.split("/")[1:])
        extra_headers = {"Title": f"{project} {name}"}
        self.run_name = name.split("/")[-1]
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                run = logger.experiment
                url = run.get_url()
                extra_headers["Click"] = url
                return extra_headers
