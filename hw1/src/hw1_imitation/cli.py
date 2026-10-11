"""Hydra is the local configuration frontend; training imports stay lazy."""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from hw1_imitation.config_composition import experiment_from_hydra, register_defaults


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def _main(config: DictConfig) -> None:
    experiment = experiment_from_hydra(config)
    from hw1_imitation.run_setup import launch_experiment

    launch_experiment(experiment, overrides=HydraConfig.get().overrides.task)


def main() -> None:
    register_defaults()
    _main()


if __name__ == "__main__":
    main()
