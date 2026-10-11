"""Run setup keeps configuration, lineage, and directory ownership explicit."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch, sentinel

import pytest
from hw1_imitation import train
from hw1_imitation.config import (
    ExperimentConfig,
    RunConfig,
    TrainConfig,
    experiment_config_from_mapping,
)
from hw1_imitation.run_setup import launch_experiment


def test_launch_saves_requested_config_and_overrides_before_training(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(run=RunConfig(name="example"))
    overrides = ["run.name=example"]

    def observe(training, *, output_dir):
        assert training is config.train
        assert output_dir == tmp_path / "exp" / "example"
        saved = json.loads((output_dir / "config.json").read_text())
        assert experiment_config_from_mapping(saved) == config
        assert json.loads((output_dir / "overrides.json").read_text()) == overrides
        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert manifest["parent_checkpoint"] is None
        assert manifest["invocation_dir"] == str(tmp_path)
        assert manifest["output_dir"] == str(output_dir)
        assert Path.cwd() == tmp_path
        return sentinel.training_result

    with patch.object(train, "run_training", side_effect=observe) as run:
        result = launch_experiment(config, overrides=overrides)
    run.assert_called_once()
    assert result is sentinel.training_result


def test_named_run_collision_preserves_existing_outputs(tmp_path):
    output = tmp_path / "same"
    output.mkdir()
    (output / "config.json").write_text("existing")
    config = ExperimentConfig(run=RunConfig(output_root=tmp_path, name="same"))
    with patch.object(train, "run_training") as run:
        with pytest.raises(FileExistsError):
            launch_experiment(config)
    assert (output / "config.json").read_text() == "existing"
    run.assert_not_called()


def test_unnamed_runs_get_distinct_directories(tmp_path):
    config = ExperimentConfig(run=RunConfig(output_root=tmp_path))
    with patch.object(train, "run_training") as run:
        launch_experiment(config)
        launch_experiment(config)
    first, second = (call.kwargs["output_dir"] for call in run.call_args_list)
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_warm_start_loads_preserved_parent_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parent = Path("parent.pt")
    parent.write_bytes(b"original checkpoint")
    config = ExperimentConfig(
        train=TrainConfig(model=None, warm_start=parent),
        run=RunConfig(output_root=tmp_path / "runs", name="child"),
    )

    def observe(training, *, output_dir):
        parent.unlink()  # Retention in the parent run cannot break this run.
        assert training == replace(
            config.train, warm_start=output_dir / "initial_policy.pt"
        )
        assert training.warm_start.read_bytes() == b"original checkpoint"
        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert manifest["parent_checkpoint"] == {
            "source": str(tmp_path / "parent.pt"),
            "snapshot": "initial_policy.pt",
            "sha256": hashlib.sha256(b"original checkpoint").hexdigest(),
        }
        requested = json.loads((output_dir / "config.json").read_text())
        assert requested["train"]["warm_start"] == "parent.pt"

    with patch.object(train, "run_training", side_effect=observe) as run:
        launch_experiment(config)
    run.assert_called_once()
    assert config.train.warm_start == parent


def test_failed_training_keeps_setup_for_diagnosis(tmp_path):
    config = ExperimentConfig(run=RunConfig(output_root=tmp_path, name="failed"))
    with patch.object(train, "run_training", side_effect=RuntimeError("failed")):
        with pytest.raises(RuntimeError, match="failed"):
            launch_experiment(config)
    assert (tmp_path / "failed" / "config.json").is_file()


def test_missing_parent_never_starts_training(tmp_path):
    config = ExperimentConfig(
        train=TrainConfig(model=None, warm_start=tmp_path / "missing.pt"),
        run=RunConfig(output_root=tmp_path / "runs"),
    )
    with patch.object(train, "run_training") as run:
        with pytest.raises(FileNotFoundError):
            launch_experiment(config)
    run.assert_not_called()


def test_core_requires_a_prepared_directory(tmp_path):
    with patch.object(train, "download_pusht") as download:
        with pytest.raises(ValueError, match="existing directory"):
            train.run_training(TrainConfig(), output_dir=tmp_path / "missing")
    download.assert_not_called()
