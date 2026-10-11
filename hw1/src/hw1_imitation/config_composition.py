"""Hydra composition ends at a concrete, validated ExperimentConfig."""

from collections.abc import Sequence

from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from hw1_imitation.config import (
    ExperimentConfig,
    config_to_dict,
    experiment_config_from_mapping,
)


def register_defaults() -> None:
    ConfigStore.instance().store(
        name="experiment_defaults", node=config_to_dict(ExperimentConfig())
    )


def experiment_from_hydra(config: DictConfig) -> ExperimentConfig:
    values = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(values, dict) or set(values) != {"train", "run"}:
        raise ValueError("Experiment configuration must contain train and run sections")
    return experiment_config_from_mapping(values)


def compose_experiment(overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Compose packaged presets without changing cwd, logging, or output files."""
    register_defaults()
    with initialize_config_module(
        version_base="1.3", config_module="hw1_imitation.conf"
    ):
        return experiment_from_hydra(
            compose(config_name="config", overrides=list(overrides))
        )
