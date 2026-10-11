"""Hydra composes concrete configs without taking over the training runtime."""

import logging
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import MissingMandatoryValue

from hw1_imitation import hydra_train
from hw1_imitation.config import TrainConfig
from hw1_imitation.config_composition import compose_training_config


def test_default_composition_tracks_python_defaults_without_state_leaks():
    assert compose_training_config() == TrainConfig()
    compose_training_config(["policy=flow", "train.lr=0.001"])
    assert compose_training_config() == TrainConfig()


def test_components_experiment_and_cli_override_precedence():
    baseline = compose_training_config(["experiment=mse_baseline"])
    assert baseline == replace(TrainConfig(), exp_name="mse-baseline")

    overridden = compose_training_config(
        [
            "experiment=mse_baseline",
            "policy=flow",
            "architecture=mlp512",
            "logging=local",
            "train.exp_name=custom",
            "train.lr=0.001",
            "train.batch_size=64",
        ]
    )
    assert overridden == replace(
        TrainConfig(),
        policy_type="flow",
        hidden_dims=(512, 512, 512),
        log_wandb=False,
        exp_name="custom",
        lr=0.001,
        batch_size=64,
    )


def test_interpolations_resolve_before_concrete_validation():
    config = compose_training_config(
        [
            "train.eval_episodes=5",
            "train.final_eval_episodes=${train.eval_episodes}",
            "train.hidden_dims=[32,16]",
            "train.data_dir=../datasets",
            "train.init_from=checkpoints/policy.pt",
        ]
    )
    assert config.eval_episodes == config.final_eval_episodes == 5
    assert config.hidden_dims == (32, 16)
    assert config.data_dir == Path("../datasets")
    assert config.init_from == Path("checkpoints/policy.pt")


@pytest.mark.parametrize(
    "overrides,error",
    [
        (["train.batch_szie=64"], ConfigCompositionException),
        (["+train.batch_szie=64"], ValueError),
        (["+unexpected=true"], ValueError),
        (["~train"], ValueError),
        (["train=null"], ValueError),
        (["train.lr=???"], MissingMandatoryValue),
        (["train.batch_size=true"], ValueError),
        (["train.policy_type=unknown"], ValueError),
    ],
)
def test_invalid_compositions_fail_before_training(overrides, error):
    with pytest.raises(error):
        compose_training_config(overrides)


def test_cli_dispatch_preserves_cwd_paths_logging_and_output_ownership(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hydra_train",
            "experiment=mse_baseline",
            "train.data_dir=relative-data",
            "train.init_from=relative.pt",
            "train.log_wandb=false",
        ],
    )

    def observe(config):
        assert Path.cwd() == tmp_path
        assert isinstance(config, TrainConfig)
        assert config.data_dir == Path("relative-data")
        assert config.init_from == Path("relative.pt")
        logging.warning("training warning remains visible")

    with patch.object(hydra_train, "run_training", side_effect=observe) as run:
        hydra_train.main()

    run.assert_called_once()
    assert run.call_args.args[0].exp_name == "mse-baseline"
    assert not run.call_args.args[0].log_wandb
    assert "training warning remains visible" in caplog.text
    assert Path.cwd() == tmp_path
    assert list(tmp_path.iterdir()) == []


def test_cli_config_inspection_does_not_start_training(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["hydra_train", "policy=flow", "--cfg", "job", "--resolve"]
    )
    with patch.object(hydra_train, "run_training") as run:
        hydra_train.main()
    run.assert_not_called()
    assert "policy_type: flow" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_configs_are_package_resources_and_compose_outside_checkout(
    tmp_path, monkeypatch
):
    root = files("hw1_imitation.conf")
    assert root.joinpath("config.yaml").is_file()
    assert root.joinpath("experiment", "mse_baseline.yaml").is_file()
    monkeypatch.chdir(tmp_path)
    assert compose_training_config(["policy=flow"]).policy_type == "flow"
    assert list(tmp_path.iterdir()) == []
