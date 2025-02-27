import logging
import os

import hydra
from datasets import Dataset, DatasetDict, load_from_disk
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from src.dataset_processor import DatasetProcessor

log = logging.getLogger(__name__)


class LichessDataModule(LightningDataModule):

    hf_dataset: DatasetDict = None
    train_split: Dataset = None
    val_split: Dataset = None
    test_split: Dataset = None
    config: DictConfig

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.tokenizer = hydra.utils.instantiate(self.config.llm.tokenizer.instance)
        self.save_to = config.dataset.save_to
        self.batch_size = config.run.loader.batch_size
        self.num_workers = config.run.loader.num_workers

    def prepare_data(self) -> Dataset:
        if os.path.exists(self.save_to):
            log.info(f"Prepared dataset already exists at {self.save_to}")
            return

        log.info("Downloading and processing dataset...")
        dataset: Dataset = hydra.utils.call(self.config.dataset.load_dataset)

        # apply dataset transforms
        transforms: list = self.config.dataset.get("transforms", [])
        processor = DatasetProcessor(transforms).with_tokenizer(self.tokenizer)
        dataset = processor.process(dataset)

        assert len(dataset["train"]) > 0

        dataset.save_to_disk(self.save_to, max_shard_size="50MB")

    def setup(self, stage: str = None):
        if not self.hf_dataset:
            self.hf_dataset = load_from_disk(self.save_to)
            self.hf_dataset = self.hf_dataset.select_columns("input_ids").with_format("torch")
            self.train_split = self.hf_dataset["train"]
            self.val_split = self.hf_dataset["test"]

    def train_dataloader(self):
        loader = DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=SimpleDataCollator(self.tokenizer),
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=SimpleDataCollator(self.tokenizer),
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=SimpleDataCollator(self.tokenizer),
        )
        return loader


class SimpleDataCollator:

    tokenizer: PreTrainedTokenizerFast
    max_length: int

    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """Pads input_ids to the max length of the sequence or the max length of the model,
        whichever is lesser.

        batch (list[dict[str, torch.Tensor]]): Batch inputs have a single key: "input_ids". When
        padding is applied, "attention_mask" is added.
        """

        padded = self.tokenizer.pad(batch)  # resulting keys: ['input_ids', 'attention_mask']
        truncated = {k: v[..., : self.max_length] for (k, v) in padded.items()}
        return truncated
