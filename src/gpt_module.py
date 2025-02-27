import logging

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from torch.optim import AdamW

log = logging.getLogger(__name__)


class GPT2Lightning(L.LightningModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.automatic_optimization = True

        self.model = self._init_llm()

        self.lr = config.lr

        self.tokenizer = hydra.utils.instantiate(self.config.llm.tokenizer.instance)

        self.save_hyperparameters()

    def _init_llm(self):
        llm = (
            hydra.utils.call(self.config.llm.from_pretrained)
            if "from_pretrained" in self.config.llm
            else hydra.utils.instantiate(self.config.llm.instance)
        ).train()

        if "loss_fn" in self.config:
            llm.loss_function = hydra.utils.instantiate(self.config.loss_fn)

        dtype_mapping = {
            "float32": torch.float32,
            "float": torch.float,  # Alias for float32
            "float16": torch.float16,
            "half": torch.half,  # Alias for float16
            "bfloat16": torch.bfloat16,
        }
        llm_dtype = llm.config.torch_dtype
        train_dtype = self.config.run.get("force_dtype", llm_dtype)

        # oddly, the LLM in not loaded to the correct dtype automatically
        llm.to(dtype_mapping[train_dtype])

        log.info(llm)
        log.info(f"LLM dtype set to {train_dtype}.")

        if "compile" in self.config:
            options = self.config.compile.options
            llm = torch.compile(llm, options=options)

        return llm

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, inputs: torch.LongTensor, attention_mask: torch.LongTensor = None):
        return self.model.forward(**inputs, labels=inputs["input_ids"])

    def training_step(self, batch, batch_idx):
        output = self.model.forward(**batch, labels=batch["input_ids"])
        loss = output.loss
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self.model.forward(**batch, labels=batch["input_ids"])
        loss = output.loss
        self.log("val_loss", loss, sync_dist=True)  # synch for multi-gpu training
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        output = self.model.forward(**batch, labels=batch["input_ids"])
        loss = output.loss
        self.log("test_loss", loss, sync_dist=True)  # synch for multi-gpu training
        return {"loss": loss}
