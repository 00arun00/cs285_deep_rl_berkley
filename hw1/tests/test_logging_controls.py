"""Exercise output controls without network access or environment rollouts."""

import csv
from unittest.mock import patch

import numpy as np
import pytest
from hw1_imitation import train
from hw1_imitation.checkpoint import load_policy
from hw1_imitation.data import Episode, EpisodesDataset
from hw1_imitation.evaluation import EvaluationResults

from .summary_helpers import summary_row_numbers, summary_row_value


class TestLoggingControls:
    def test_cli_defaults_and_disable_flags(self):
        defaults = train.parse_train_config([])
        for field in ("log_csv", "log_wandb", "save_checkpoints", "show_summary"):
            assert getattr(defaults, field)
            parsed = train.parse_train_config(["--no-" + field.replace("_", "-")])
            for other in ("log_csv", "log_wandb", "save_checkpoints", "show_summary"):
                assert getattr(parsed, other) == (other != field)

    def test_disabled_logger_skips_video_validation(self, tmp_path):
        logger = train.ExperimentLogger(tmp_path, log_csv=False)
        logger.log_eval(
            global_step=1,
            mean_reward=0.5,
            num_episodes=1,
            video_paths=(tmp_path / "missing.mp4",),
        )
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("log_csv", [False, True], ids=["log_csv=off", "log_csv=on"])
@pytest.mark.parametrize(
    "log_wandb", [False, True], ids=["log_wandb=off", "log_wandb=on"]
)
@pytest.mark.parametrize(
    "save_checkpoints",
    [False, True],
    ids=["save_checkpoints=off", "save_checkpoints=on"],
)
@pytest.mark.parametrize(
    "show_summary", [False, True], ids=["show_summary=off", "show_summary=on"]
)
def test_all_output_combinations(
    log_csv, log_wandb, save_checkpoints, show_summary, tmp_path, capsys, monkeypatch
):
    # Keep semantic rows on one line regardless of the invoking terminal.
    monkeypatch.setenv("COLUMNS", "160")
    episode_count, episode_length = 5, 2
    rng = np.random.default_rng(7)
    episodes = EpisodesDataset(
        tuple(
            Episode(
                i,
                rng.normal(size=(episode_length, 3)).astype(np.float32),
                rng.normal(size=(episode_length, 2)).astype(np.float32),
            )
            for i in range(episode_count)
        )
    )
    root = tmp_path
    config = train.TrainConfig(
        log_csv=log_csv,
        log_wandb=log_wandb,
        save_checkpoints=save_checkpoints,
        show_summary=show_summary,
        hidden_dims=(4,),
        chunk_size=1,
        num_epochs=1,
        batch_size=8,
        num_video_episodes=0,
    )
    real_step = train.train_step
    real_validation = train.compute_validation_loss
    observed = {}

    def step(*args, **kwargs):
        result = real_step(*args, **kwargs)
        observed["train_loss"] = result.item()
        return result

    def validate(*args, **kwargs):
        result = real_validation(*args, **kwargs)
        observed["validation_loss"] = result.loss_mean
        observed["validation_examples"] = result.examples_count
        return result

    with (
        patch.object(train, "train_step", side_effect=step),
        patch.object(train, "compute_validation_loss", side_effect=validate),
        patch.object(train, "LOGDIR_PREFIX", str(root)),
        patch.object(train, "download_pusht", return_value=root),
        patch.object(train, "load_episodes_dataset", return_value=episodes),
        patch.object(
            train.torch.accelerator,
            "current_accelerator",
            return_value=None,
        ),
        patch.object(train.wandb, "init") as initialize,
        patch.object(train.wandb, "Artifact") as artifact,
        patch.object(
            train,
            "evaluate_policy",
            return_value=EvaluationResults(0.5, 1, ()),
        ) as evaluate,
    ):
        run = initialize.return_value.__enter__.return_value
        run.offline = False
        run.id = "test"
        train.run_training(config)

    captured = capsys.readouterr()
    if show_summary:
        # The five equal-length episodes split 80/20 into four train and one
        # validation episode. With horizon one, every timestep is a sample.
        train_episodes = episode_count * 4 // 5
        validation_episodes = episode_count - train_episodes
        train_samples = train_episodes * episode_length
        validation_samples = validation_episodes * episode_length
        optimizer_steps = config.num_epochs * (train_samples // config.batch_size)
        assert summary_row_numbers(captured.err, "Episodes") == [
            train_episodes,
            validation_episodes,
        ]
        assert summary_row_numbers(captured.err, "Training samples") == [train_samples]
        assert summary_row_numbers(captured.err, "Validation samples") == [
            validation_samples
        ]
        assert summary_row_numbers(captured.err, "Training budget") == [
            config.num_epochs,
            optimizer_steps,
        ]
        assert summary_row_value(captured.err, "Device") == "cpu"
        assert "Training samples" not in captured.out
    else:
        assert "Training samples" not in captured.err + captured.out

    evaluate.assert_called_once()
    train_metrics = {
        "epoch": 0,
        "loss_window_mean": observed["train_loss"],
        "window_examples_count": 8,
    }
    validation_metrics = {
        "loss_mean": observed["validation_loss"],
        "examples_count": observed["validation_examples"],
    }
    assert observed["validation_examples"] == 2
    eval_metrics = {"mean_reward": 0.5, "num_episodes": 1}
    expected = {
        "train": train_metrics,
        "validation": validation_metrics,
        "eval": eval_metrics,
    }
    if log_wandb:
        initialize.assert_called_once()
        payloads = [call.args[0] for call in run.log.call_args_list]
        assert payloads == [
            {
                "global_step": 1,
                **{f"{namespace}/{key}": value for key, value in metrics.items()},
            }
            for namespace, metrics in expected.items()
        ]
    else:
        initialize.assert_not_called()
        run.log.assert_not_called()

    csv_paths = sorted(root.rglob("*.csv"))
    assert {path.name for path in csv_paths} == (
        {"train.csv", "validation.csv", "eval.csv"} if log_csv else set()
    )
    for path in csv_paths:
        with path.open() as file:
            rows = list(csv.DictReader(file))
        expected_row = {
            "global_step": "1",
            **{key: str(value) for key, value in expected[path.stem].items()},
        }
        if path.stem == "eval":
            expected_row["video_paths"] = "[]"
        assert rows == [expected_row]

    checkpoints = list(root.rglob("*.pt"))
    assert len(checkpoints) == int(save_checkpoints)
    if save_checkpoints:
        assert checkpoints[0].parent.name == "checkpoints"
        load_policy(checkpoints[0])
    else:
        assert list(root.rglob("checkpoints")) == []

    if log_wandb and save_checkpoints:
        artifact.assert_called_once()
        run.log_artifact.assert_called_once()
        run.log_artifact.return_value.wait.assert_called_once()
    else:
        artifact.assert_not_called()
        run.log_artifact.assert_not_called()
