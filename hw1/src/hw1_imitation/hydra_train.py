"""Optional Hydra CLI; the original train.py Tyro CLI remains available."""

import hydra
from omegaconf import DictConfig

from hw1_imitation.config_composition import (
    register_training_defaults,
    training_config_from_hydra,
)
from hw1_imitation.train import run_training


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def _train(config: DictConfig) -> None:
    run_training(training_config_from_hydra(config))


def main() -> None:
    register_training_defaults()
    _train()


if __name__ == "__main__":
    main()
