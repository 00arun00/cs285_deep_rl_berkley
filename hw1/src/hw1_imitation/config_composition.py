"""Compose Hydra presets into the existing concrete training configuration."""

from collections.abc import Sequence

from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from hw1_imitation.config import (
    TrainConfig,
    config_to_dict,
    train_config_from_mapping,
)


def register_training_defaults() -> None:
    """Keep Python defaults authoritative without freezing the composition tree."""
    ConfigStore.instance().store(
        name="training_defaults",
        node={"train": config_to_dict(TrainConfig())},
    )


def training_config_from_hydra(config: DictConfig) -> TrainConfig:
    """Resolve interpolations and validate the complete application configuration."""
    values = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(values, dict) or set(values) != {"train"}:
        raise ValueError("Training configuration must contain only a 'train' mapping")
    if not isinstance(values["train"], dict):
        raise ValueError("Training configuration must contain a 'train' mapping")
    return train_config_from_mapping(values["train"])


def compose_training_config(overrides: Sequence[str] = ()) -> TrainConfig:
    """Compose packaged presets without changing cwd, logging, or output files."""
    register_training_defaults()
    with initialize_config_module(
        version_base="1.3", config_module="hw1_imitation.conf"
    ):
        config = compose(config_name="config", overrides=list(overrides))
        return training_config_from_hydra(config)
