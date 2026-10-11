"""Hydra is the local configuration frontend; training imports stay lazy."""

import hydra
from omegaconf import DictConfig

from hw1_imitation.config_composition import experiment_from_hydra, register_defaults


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def _main(config: DictConfig) -> None:
    experiment = experiment_from_hydra(config)
    from hw1_imitation.train import run_training

    run_training(
        experiment.train,
        output_root=experiment.run.output_root,
        run_name=experiment.run.name,
    )


def main() -> None:
    register_defaults()
    _main()


if __name__ == "__main__":
    main()
