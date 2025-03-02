import os

import hydra
from datasets import Dataset, DatasetDict, load_from_disk
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from src.dataset_processor import DatasetProcessor


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
            print(f"Prepared dataset already exists at {self.save_to}")
            return

        print("Downloading and processing dataset...")
        dataset: Dataset = hydra.utils.call(self.config.dataset.load_dataset)

        # apply dataset transforms
        transforms: list = self.config.dataset.get("transforms", [])
        processor = DatasetProcessor(transforms).with_tokenizer(self.tokenizer)
        dataset = processor.process(dataset)

        assert len(dataset["train"]) > 0

        dataset.save_to_disk(
            self.save_to,
            max_shard_size="50MB",
            num_proc=self.config.run.hf_dataset_num_proc,
        )

    def setup(self, stage: str = None):
        if not self.hf_dataset:
            self.hf_dataset = load_from_disk(self.save_to)
            self.train_split = self.hf_dataset["train"]
            self.val_split = self.hf_dataset["test"]

    def train_dataloader(self):
        loader = DataLoader(
            self.train_split,
            shuffle=True,
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

    def __call__(self, batch_text_or_text_pairs: list[str]):
        texts = [b["text"] for b in batch_text_or_text_pairs]
        try:
            encoding = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        except Exception as ex:
            print(texts)
        return encoding
