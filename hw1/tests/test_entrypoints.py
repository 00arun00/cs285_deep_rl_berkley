"""Local and Modal launchers share composition and dispatch without remote work."""

import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from hw1_imitation import cli, modal_train, run_setup
from hw1_imitation.config import ExperimentConfig


def test_hydra_cli_dispatch(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "model=flow",
            "logging=local",
            "run.name=example",
            "train.data.dir=relative",
        ],
    )

    def observe(config, **kwargs):
        assert isinstance(config, ExperimentConfig)
        assert config.train.model.policy_type == "flow"
        assert config.train.data.dir == Path("relative")
        assert config.run.output_root == Path("exp")
        assert config.run.name == "example"
        assert kwargs["overrides"] == sys.argv[1:]
        assert Path.cwd() == tmp_path
        logging.warning("visible training warning")

    with patch.object(run_setup, "launch_experiment", side_effect=observe) as run:
        cli.main()
    run.assert_called_once()
    assert "visible training warning" in caplog.text
    assert not list(tmp_path.iterdir())


def test_config_inspection_imports_no_training_dependencies(tmp_path):
    code = "import sys; sys.argv=['train','model=flow','--cfg','job','--resolve']; from hw1_imitation.cli import main; main(); assert not {'torch','modal','wandb'} & sys.modules.keys()"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "policy_type: flow" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "overrides,path,root",
    [
        ((), "/vol/data", "/vol/exp"),
        (("experiment=mse_baseline",), "/vol/data", "/vol/exp"),
        (("train.data.dir=/custom", "run.output_root=/output"), "/custom", "/output"),
    ],
)
def test_modal_composes_then_trains_and_commits(overrides, path, root):
    events = []

    def observe(config, **kwargs):
        assert config.train.data.dir == Path(path)
        assert config.run.output_root == Path(root)
        assert kwargs["overrides"] == ["runtime=modal", *overrides]
        events.append("train")

    with (
        patch.object(run_setup, "launch_experiment", side_effect=observe),
        patch.object(
            modal_train.volume, "commit", side_effect=lambda: events.append("commit")
        ),
    ):
        modal_train.train_remote.local(*overrides)
    assert events == ["train", "commit"]
    assert list(modal_train.app.registered_functions) == ["train_remote"]


@pytest.mark.parametrize("failure", ["configuration", "training"])
def test_modal_failure_does_not_commit(failure):
    with (
        patch.object(
            run_setup, "launch_experiment", side_effect=RuntimeError("training failed")
        ) as run,
        patch.object(modal_train.volume, "commit") as commit,
    ):
        with pytest.raises((ValueError, RuntimeError)):
            modal_train.train_remote.local(
                *(["train.optimization.lr=-1"] if failure == "configuration" else [])
            )
    assert run.call_count == (failure == "training")
    commit.assert_not_called()
