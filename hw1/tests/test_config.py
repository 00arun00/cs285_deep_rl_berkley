"""The composition boundary produces the existing concrete training contract."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from hw1_imitation import config, train


def test_legacy_imports_and_default_construction_remain_compatible():
    assert train.TrainConfig is config.TrainConfig
    assert train.parse_train_config is config.parse_train_config
    assert train.config_to_dict is config.config_to_dict
    assert config.train_config_from_mapping({}) == train.parse_train_config([])
    assert config.train_config_from_mapping({}).data_dir == Path("data")


def test_mapping_converts_types_without_rewriting_paths_or_checkpoint_settings():
    values = {
        "data_dir": "../data",
        "init_from": "checkpoints/policy.pt",
        "policy_type": "flow",
        "hidden_dims": [64, 32],
        "video_size": [320, 240],
        "batch_size": "64",
        "lr": "0.001",
        "seed": 2**100,
        "exp_name": None,
        "log_wandb": False,
    }
    result = config.train_config_from_mapping(values)

    assert result == config.TrainConfig(
        data_dir=Path("../data"),
        init_from=Path("checkpoints/policy.pt"),
        policy_type="flow",
        hidden_dims=(64, 32),
        video_size=(320, 240),
        batch_size=64,
        lr=0.001,
        seed=2**100,
        log_wandb=False,
    )
    assert values["hidden_dims"] == [64, 32]
    assert values["data_dir"] == "../data"
    with pytest.raises(FrozenInstanceError):
        result.lr = 0.1


@pytest.mark.parametrize(
    "values",
    [
        {"batch_szie": 64},
        {"policy_type": "unknown"},
        {"batch_size": True},
        {"lr": False},
        {"seed": True},
        {"eval_env_variation": False},
        {"hidden_dims": [64, True]},
        {"video_size": [256, False]},
        {"video_size": [256]},
        {"video_size": [256, 256, 256]},
        {"num_epochs": 0},
        {"batch_size": 1.5},
        {"checkpoint_top_k": -1},
        {"eval_env_variation": 2**32},
        {"seed": -1},
        {"num_video_episodes": -1},
        {"lr": float("nan")},
        {"lr": "inf"},
        {"lr": -0.01},
        {"weight_decay": float("-inf")},
        {"weight_decay": -0.01},
    ],
)
def test_invalid_mappings_fail_before_training(values):
    with pytest.raises(ValueError):
        config.train_config_from_mapping(values)


def test_supported_edge_values_and_legacy_serialization_are_preserved():
    result = config.train_config_from_mapping(
        {
            "hidden_dims": [],
            "lr": 0,
            "weight_decay": 0,
            "num_video_episodes": 0,
            "video_size": [1, 1],
            "init_from": None,
        }
    )
    assert result.hidden_dims == ()
    assert result.video_size == (1, 1)
    serialized = config.config_to_dict(result)
    assert serialized["data_dir"] == "data"
    assert serialized["init_from"] is None
    assert serialized["hidden_dims"] == ()
    assert config.train_config_from_mapping(serialized) == result


def test_tyro_custom_defaults_keep_modal_data_directory_and_overrides():
    defaults = config.TrainConfig(data_dir=Path("/vol/data"))
    result = train.parse_train_config(
        ["--num-epochs", "20", "--no-log-wandb"],
        defaults=defaults,
        description="Train on Modal.",
    )
    assert result.data_dir == Path("/vol/data")
    assert result.num_epochs == 20
    assert not result.log_wandb
    assert defaults == config.TrainConfig(data_dir=Path("/vol/data"))
