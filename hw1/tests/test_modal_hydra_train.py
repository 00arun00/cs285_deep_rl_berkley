"""Exercise Modal entry points locally with training and volume I/O mocked."""

from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from unittest.mock import Mock, call

import pytest
from hydra.errors import ConfigCompositionException

from hw1_imitation import modal_hydra_train, modal_train
from hw1_imitation.config import TrainConfig


def test_hydra_app_has_one_entry_point_and_reuses_modal_resources():
    assert modal_hydra_train.app is not modal_train.app
    assert modal_hydra_train.app.name == "hw1-imitation-hydra"
    assert set(modal_hydra_train.app.registered_functions) == {"train_remote"}
    assert set(modal_train.app.registered_functions) == {"train_remote"}
    assert modal_hydra_train.image is modal_train.image
    assert modal_hydra_train.env is modal_train.env
    assert modal_hydra_train.volume is modal_train.volume

    # Compare the SDK's registered resources, not just the imported constants.
    original = modal_train.train_remote.spec
    composed = modal_hydra_train.train_remote.spec
    assert composed.image is original.image
    assert composed.volumes == original.volumes
    assert composed.cpu == original.cpu
    assert composed.gpus == original.gpus


@pytest.mark.parametrize(
    "overrides,changes",
    [
        ([], {}),
        (["experiment=mse_baseline"], {"exp_name": "mse-baseline"}),
        (
            [
                "experiment=mse_baseline",
                "policy=flow",
                "train.lr=0.001",
                "train.data_dir=/vol/custom-data",
                "train.hidden_dims=[32,16]",
            ],
            {
                "exp_name": "mse-baseline",
                "policy_type": "flow",
                "lr": 0.001,
                "data_dir": Path("/vol/custom-data"),
                "hidden_dims": (32, 16),
            },
        ),
        (["data=pusht"], {"data_dir": Path("data")}),
    ],
)
def test_modal_composes_packaged_config_then_trains_and_commits(
    overrides, changes, tmp_path, monkeypatch
):
    assert files("hw1_imitation.conf").joinpath("data", "modal.yaml").is_file()
    monkeypatch.chdir(tmp_path)
    events = Mock()
    monkeypatch.setattr(modal_hydra_train, "run_training", events.train)
    monkeypatch.setattr(modal_hydra_train.volume, "commit", events.commit)

    # local() runs only the Python function; no image build or cloud invocation.
    with pytest.warns(UserWarning, match="executing locally"):
        modal_hydra_train.train_remote.local(*overrides)

    expected = replace(TrainConfig(data_dir=Path("/vol/data")), **changes)
    assert events.mock_calls == [call.train(expected), call.commit()]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "overrides,error",
    [
        (["train.batch_szie=64"], ConfigCompositionException),
        (["train.batch_size=true"], ValueError),
    ],
)
def test_invalid_configuration_never_starts_training_or_commits(
    overrides, error, monkeypatch
):
    events = Mock()
    monkeypatch.setattr(modal_hydra_train, "run_training", events.train)
    monkeypatch.setattr(modal_hydra_train.volume, "commit", events.commit)
    with (
        pytest.warns(UserWarning, match="executing locally"),
        pytest.raises(error),
    ):
        modal_hydra_train.train_remote.local(*overrides)
    assert events.mock_calls == []


def test_training_failure_propagates_without_committing(monkeypatch):
    events = Mock()
    events.train.side_effect = RuntimeError("training failed")
    monkeypatch.setattr(modal_hydra_train, "run_training", events.train)
    monkeypatch.setattr(modal_hydra_train.volume, "commit", events.commit)

    with (
        pytest.warns(UserWarning, match="executing locally"),
        pytest.raises(RuntimeError, match="training failed"),
    ):
        modal_hydra_train.train_remote.local()
    assert events.mock_calls == [call.train(TrainConfig(data_dir=Path("/vol/data")))]


def test_original_modal_entry_point_still_accepts_tyro_flags(monkeypatch):
    events = Mock()
    monkeypatch.setattr(modal_train, "run_training", events.train)
    monkeypatch.setattr(modal_train.volume, "commit", events.commit)

    with pytest.warns(UserWarning, match="executing locally"):
        modal_train.train_remote.local("--num-epochs", "20", "--no-log-wandb")
    assert events.mock_calls == [
        call.train(
            TrainConfig(data_dir=Path("/vol/data"), num_epochs=20, log_wandb=False)
        ),
        call.commit(),
    ]
