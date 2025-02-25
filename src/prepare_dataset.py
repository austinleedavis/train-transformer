import logging
import os

import datasets
import hydra
import pyrootutils
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerFast

from dataset_processor import DatasetProcessor

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
    tokenizer = hydra.utils.instantiate(cfg.llm.tokenizer.instance)
    dataset = DatasetLoader(cfg, tokenizer).load()


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
