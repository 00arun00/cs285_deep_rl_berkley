"""Exercise output controls without network access or environment rollouts."""

import csv
from unittest.mock import patch

import numpy as np
import pytest
from hw1_imitation import train
from hw1_imitation.checkpoint import load_policy
from hw1_imitation.data import Episode, EpisodesDataset
from hw1_imitation.evaluation import EvaluationResults


class TestLoggingControls:
    def test_cli_defaults_and_disable_flags(self):
        defaults = train.parse_train_config([])
        for field in ("log_csv", "log_wandb", "save_checkpoints"):
            assert getattr(defaults, field)
            parsed = train.parse_train_config(["--no-" + field.replace("_", "-")])
            for other in ("log_csv", "log_wandb", "save_checkpoints"):
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
    log_csv, log_wandb, save_checkpoints, show_summary, tmp_path
):
    rng = np.random.default_rng(7)
    episodes = EpisodesDataset(
        tuple(
            Episode(
                i,
                rng.normal(size=(2, 3)).astype(np.float32),
                rng.normal(size=(2, 2)).astype(np.float32),
            )
            for i in range(5)
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
    with (
        patch.object(train, "LOGDIR_PREFIX", str(root)),
        patch.object(train, "download_pusht", return_value=root),
        patch.object(train, "load_episodes_dataset", return_value=episodes),
        patch.object(
            train.torch.accelerator,
            "current_accelerator",
            return_value=None,
        ),
        patch.object(train, "Console") as console,
        patch.object(
            train,
            "build_training_summary",
            wraps=train.build_training_summary,
        ) as build_summary,
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

    if config.show_summary:
        console.assert_called_once_with(stderr=True)
        console.return_value.print.assert_called_once()
        summary = build_summary.call_args.kwargs
        assert summary["config"] is config
        assert summary["train_episodes"] == 4
        assert summary["validation_episodes"] == 1
        assert summary["train_samples"] == 8
        assert summary["validation_samples"] == 2
        assert summary["steps_per_epoch"] == 1
        assert summary["device"] == "cpu"
    else:
        console.assert_not_called()
        build_summary.assert_not_called()

    evaluate.assert_called_once()
    if log_wandb:
        initialize.assert_called_once()
        assert run.log.call_count == 3
        run.log.assert_any_call(
            {
                "global_step": 1,
                "eval/mean_reward": 0.5,
                "eval/num_episodes": 1,
            }
        )
    else:
        initialize.assert_not_called()
        run.log.assert_not_called()

    csv_paths = sorted(root.rglob("*.csv"))
    assert len(csv_paths) == (3 if log_csv else 0)
    for path in csv_paths:
        with path.open() as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == 1
        assert rows[0]["global_step"] == "1"
        if path.name == "eval.csv":
            assert rows[0] == {
                "global_step": "1",
                "mean_reward": "0.5",
                "num_episodes": "1",
                "video_paths": "[]",
            }

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
