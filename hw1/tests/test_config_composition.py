"""Composition preserves nested types, override order, and process state."""

from importlib.resources import files
from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import MissingMandatoryValue

from hw1_imitation.config import ExperimentConfig
from hw1_imitation.config_composition import compose_experiment


def test_defaults_and_repeated_composition():
    assert compose_experiment() == ExperimentConfig()
    compose_experiment(["model=flow", "train.optimization.lr=0.001"])
    assert compose_experiment() == ExperimentConfig()


def test_experiment_component_and_value_precedence():
    cfg = compose_experiment(
        [
            "experiment=mse_baseline",
            "model=flow",
            "architecture=mlp512",
            "logging=local",
            "train.optimization.lr=0.001",
            "run.name=custom",
        ]
    )
    assert cfg.train.model.policy_type == "flow"
    assert cfg.train.model.hidden_dims == (512, 512, 512)
    assert cfg.train.optimization.lr == 0.001
    assert not cfg.train.logging.wandb
    assert cfg.run.name == "custom"


def test_interpolations_and_concrete_types():
    cfg = compose_experiment(
        [
            "train.evaluation.episodes=5",
            "train.evaluation.final_episodes=${train.evaluation.episodes}",
            "train.data.dir=../data",
            "train.model.hidden_dims=[32,16]",
        ]
    )
    assert cfg.train.evaluation.episodes == cfg.train.evaluation.final_episodes == 5
    assert cfg.train.model.hidden_dims == (32, 16)
    assert cfg.train.data.dir == Path("../data")


def test_warm_start_preset_is_explicit():
    cfg = compose_experiment(
        ["initialization=checkpoint", "train.warm_start=parent.pt"]
    )
    assert cfg.train.model is None
    assert cfg.train.warm_start == Path("parent.pt")
    with pytest.raises(ValueError, match="Choose model"):
        compose_experiment(["initialization=checkpoint"])
    with pytest.raises(ValueError):
        compose_experiment(["train.warm_start=parent.pt"])


def test_modal_defaults_and_user_paths_survive_named_experiment():
    cfg = compose_experiment(["runtime=modal", "experiment=mse_baseline"])
    assert cfg.train.data.dir == Path("/vol/data")
    assert cfg.run.output_root == Path("/vol/exp")
    cfg = compose_experiment(
        ["runtime=modal", "train.data.dir=/custom", "run.output_root=/outputs"]
    )
    assert cfg.train.data.dir == Path("/custom")
    assert cfg.run.output_root == Path("/outputs")


@pytest.mark.parametrize(
    "overrides,error",
    [
        (["train.optimization.batch_szie=64"], ConfigCompositionException),
        (["+train.optimization.batch_szie=64"], ValueError),
        (["+unexpected=true"], ValueError),
        (["~train"], ValueError),
        (["train=null"], ValueError),
        (["train.optimization.lr=???"], MissingMandatoryValue),
        (["train.optimization.batch_size=true"], ValueError),
        (["train.model.policy_type=unknown"], ValueError),
    ],
)
def test_invalid_compositions(overrides, error):
    with pytest.raises(error):
        compose_experiment(overrides)


def test_packaged_configs_compose_outside_checkout(tmp_path, monkeypatch):
    root = files("hw1_imitation.conf")
    assert root.joinpath("config.yaml").is_file()
    assert root.joinpath("initialization/checkpoint.yaml").is_file()
    monkeypatch.chdir(tmp_path)
    assert compose_experiment(["model=flow"]).train.model.policy_type == "flow"
    assert not list(tmp_path.iterdir())
