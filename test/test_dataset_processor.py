import hydra
from omegaconf import DictConfig
import pyrootutils

from dataset_processor import DatasetProcessor

if __name__ == "__main__":

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

    @hydra.main(**_HYDRA_PARAMS)
    def main(cfg: DictConfig):
        # Load dataset using Hydra-instantiated parameters
        print
        dataset = hydra.utils.call(cfg.dataset.instance)

        # Process dataset
        transforms = cfg.dataset.get("transforms", [])
        processor = DatasetProcessor(transforms)
        dataset = processor.process(dataset)

        print(dataset)

    main()
