"""Nested settings validate Python and composed mappings without importing training."""

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from hw1_imitation.config import (
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    TrainConfig,
    config_to_dict,
    experiment_config_from_mapping,
    train_config_from_mapping,
)


def test_defaults_round_trip_and_freezing():
    cfg = ExperimentConfig()
    assert cfg.train.model == ModelConfig()
    assert cfg.train.optimization == OptimizationConfig()
    assert cfg.train.evaluation.episodes == cfg.train.evaluation.final_episodes == 100
    assert cfg.train.checkpoints.top_k == 3
    assert experiment_config_from_mapping(config_to_dict(cfg)) == cfg
    with pytest.raises(FrozenInstanceError):
        cfg.train.optimization.lr = 1


def test_concrete_paths_tuples_and_zero_learning_rate():
    cfg = train_config_from_mapping(
        {
            "data": {"dir": "relative/data"},
            "model": {"hidden_dims": [32, 16]},
            "optimization": {"lr": 0},
        }
    )
    assert cfg.data.dir == Path("relative/data")
    assert cfg.model.hidden_dims == (32, 16)
    assert cfg.optimization.lr == 0
    assert TrainConfig(model=ModelConfig(hidden_dims=())).model.hidden_dims == ()


@pytest.mark.parametrize(
    "values",
    [
        {"lr": 0.1},
        {"optimization": {"typo": 1}},
        {"optimization": {"lr": True}},
        {"optimization": {"lr": float("nan")}},
        {"optimization": {"lr": float("inf")}},
        {"optimization": {"weight_decay": -1}},
        {"optimization": {"batch_size": True}},
        {"optimization": {"num_epochs": 1.5}},
        {"evaluation": {"interval": 0}},
        {"evaluation": {"video_size": [255, 256]}},
        {"logging": {"wandb": "false"}},
        {"model": {"policy_type": "unknown"}},
        {"model": {"hidden_dims": [True]}},
        {"randomness": {"seed": -1}},
        {"randomness": {"eval_env_variation": 2**32}},
        {"checkpoints": {"top_k": -1}},
        {"warm_start": "parent.pt"},
        {"model": None},
    ],
)
def test_invalid_nested_input(values):
    with pytest.raises(ValueError):
        train_config_from_mapping(values)


def test_python_construction_uses_same_validation():
    with pytest.raises(ValueError):
        OptimizationConfig(lr=-1)
    with pytest.raises(ValueError):
        ModelConfig(chunk_size=True)


def test_warm_start_has_one_model_source():
    cfg = train_config_from_mapping({"model": None, "warm_start": "parent.pt"})
    assert cfg.warm_start == Path("parent.pt")
    assert cfg.model is None


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../outside", "nested/run", "nested\\run"]
)
def test_run_name_cannot_escape_output_root(name):
    with pytest.raises(ValueError):
        experiment_config_from_mapping({"run": {"name": name}})


def test_config_import_is_lightweight():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import hw1_imitation.config; assert not {'torch', 'modal', 'wandb', 'hw1_imitation.train'} & sys.modules.keys()",
        ],
        check=True,
    )
