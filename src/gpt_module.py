import logging

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from transformers.loss.loss_utils import ForCausalLMLoss


class GPT2Lightning(L.LightningModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.cfg = config
        self.config = OmegaConf.to_container(config)
        self.automatic_optimization = True

        self.model = self._init_llm()

        self.lr = self.cfg.run.lr

        self.tokenizer = hydra.utils.instantiate(self.cfg.llm.tokenizer.instance)
        self.vocab_size = self.tokenizer.vocab_size
        self.save_hyperparameters()

    def _init_llm(self):
        llm = (
            hydra.utils.call(self.cfg.llm.from_pretrained)
            if "from_pretrained" in self.cfg.llm
            else hydra.utils.instantiate(self.cfg.llm.instance)
        ).train()  # enable training mode so dropout is used
        # llm.config.loss_type = "ForCausalLMLoss"

        dtype_mapping = {
            "float32": torch.float32,
            "float": torch.float,  # Alias for float32
            "float16": torch.float16,
            "half": torch.half,  # Alias for float16
            "bfloat16": torch.bfloat16,
        }
        llm_dtype = llm.config.torch_dtype
        train_dtype = self.cfg.run.get("force_dtype", llm_dtype)

        # oddly, the LLM in not loaded to the correct dtype automatically
        llm.to(dtype_mapping[train_dtype])

        # print(llm)
        print(f"LLM dtype set to {train_dtype}.")

        if "compile" in self.cfg:
            options = self.cfg.compile.options
            llm = torch.compile(llm, options=options)

        return llm

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, inputs: torch.LongTensor, attention_mask: torch.LongTensor = None):
        return self.model.forward(**inputs)

    def training_step(self, batch, batch_idx):
        logits = self.model.forward(**batch).logits

        loss = ForCausalLMLoss(
            logits=logits,
            labels=batch["input_ids"],
            vocab_size=self.vocab_size,
        )
        self.log("train/loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        logits = self.model.forward(**batch).logits

        loss = ForCausalLMLoss(
            logits=logits,
            labels=batch["input_ids"],
            vocab_size=self.vocab_size,
        )
        self.log("val/loss", loss, sync_dist=True)  # synch for multi-gpu training
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        logits = self.model.forward(**batch).logits

        loss = ForCausalLMLoss(
            logits=logits,
            labels=batch["input_ids"],
            vocab_size=self.vocab_size,
        )
        self.log("test/loss", loss, sync_dist=True)  # synch for multi-gpu training
        return {"loss": loss}
