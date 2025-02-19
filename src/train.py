import logging
import os

import datasets
import hydra
import pyrootutils
import torch
import transformers
from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

import wandb
from callbacks import NtfyCallback
from dataset_processor import DatasetProcessor

load_dotenv()

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(root / "configs"),
    "config_name": "train.yaml",
}

log = logging.getLogger(__name__)


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    trainer = TrainingManager(cfg)
    trainer.train()


class TrainingManager:

    config: DictConfig
    dataset: datasets.Dataset
    hf_trainer: Trainer
    hf_trainer_args: TrainingArguments
    llm: GPT2LMHeadModel
    tokenizer: PreTrainedTokenizerFast

    def __init__(self, config: DictConfig):
        self.config: DictConfig = config
        self.ntfy = NtfyCallback(topic=os.environ.get("NTFY_TOPIC", None))
        self.llm = self._init_llm()
        self.tokenizer = hydra.utils.instantiate(config.llm.tokenizer.instance)
        self.tokenizer.model_max_length = self.llm.config.n_ctx

        # setup and process dataset
        self.dataset = DatasetLoader(self.config, self.tokenizer).load()

        # setup Hf Trainer
        self.hf_trainer_args = hydra.utils.instantiate(config.hf_trainer_args)
        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.hf_trainer = Trainer(
            self.llm,
            args=self.hf_trainer_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=collator,
            processing_class=self.tokenizer,
            callbacks=[self.ntfy],
        )

    def train(self):
        log.info("Train!")

        class ProfCallback(transformers.TrainerCallback):
            def __init__(self, prof):
                self.prof = prof

            def on_step_end(self, args, state, control, **kwargs):
                self.prof.step()

        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    skip_first=0, wait=0, warmup=0, active=2, repeat=2
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("hf-training-trainer"),
                profile_memory=True,
                with_stack=True,
                record_shapes=True,
            ) as prof:

                self.hf_trainer.add_callback(ProfCallback(prof=prof))

                self.hf_trainer.train()
        finally:
            self.ntfy.send_notification("Training failed")
            if wandb.run:
                wandb.run.finish(exit_code=-1)

    def _init_llm(self):
        llm = (
            hydra.utils.call(self.config.llm.from_pretrained)
            if "from_pretrained" in self.config.llm
            else hydra.utils.instantiate(self.config.llm.instance)
        )
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

        return llm


class DatasetLoader:
    """Not to be confused with a dataloader, this handles loading, preparing, and cacheing the
    dataset on disk."""

    def __init__(self, config: DictConfig, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        self.tokenizer = tokenizer
        self.save_to = config.dataset.save_to

    def load(self) -> datasets.Dataset:
        if os.path.exists(self.save_to):
            log.info("Loading dataset from cache...")
            self.dataset = datasets.load_from_disk(self.save_to)
        else:
            log.info("Downloading and processing dataset...")
            self.dataset = self._download_and_process()

        log.info(self.dataset)

        return self.dataset

    def _download_and_process(self) -> datasets.Dataset:

        self.dataset: datasets.Dataset = hydra.utils.call(self.config.dataset.load_dataset)

        # apply dataset transforms
        transforms: list = self.config.dataset.get("transforms", [])
        processor = DatasetProcessor(transforms)
        self.dataset = processor.process(self.dataset)

        # tokenize the dataset
        self.dataset = self.dataset.map(lambda ex: self.tokenizer(ex["text"]), desc="Tokenize")

        assert len(self.dataset["train"]) > 0

        self.dataset.save_to_disk(self.save_to, max_shard_size="50MB")
        return self.dataset


if __name__ == "__main__":
    main(None)
