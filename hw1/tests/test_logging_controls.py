"""Exercise output controls without network access or environment rollouts."""

import csv
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hw1_imitation import train
from hw1_imitation.checkpoint import load_policy
from hw1_imitation.data import Episode, EpisodesDataset
from hw1_imitation.evaluation import EvaluationResults


class LoggingControlsTests(unittest.TestCase):
    def test_cli_defaults_and_disable_flags(self):
        defaults = train.parse_train_config([])
        for field in ("log_csv", "log_wandb", "save_checkpoints"):
            self.assertTrue(getattr(defaults, field))
            parsed = train.parse_train_config(["--no-" + field.replace("_", "-")])
            for other in ("log_csv", "log_wandb", "save_checkpoints"):
                self.assertEqual(getattr(parsed, other), other != field)

    def test_all_output_combinations(self):
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
        for log_csv, log_wandb, save_checkpoints, show_summary in itertools.product(
            (False, True), repeat=4
        ):
            with (
                self.subTest(
                    log_csv=log_csv,
                    log_wandb=log_wandb,
                    save_checkpoints=save_checkpoints,
                    show_summary=show_summary,
                ),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
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
                    self.assertIs(summary["config"], config)
                    self.assertEqual(summary["train_episodes"], 4)
                    self.assertEqual(summary["validation_episodes"], 1)
                    self.assertEqual(summary["train_samples"], 8)
                    self.assertEqual(summary["validation_samples"], 2)
                    self.assertEqual(summary["steps_per_epoch"], 1)
                    self.assertEqual(summary["device"], "cpu")
                else:
                    console.assert_not_called()
                    build_summary.assert_not_called()

                evaluate.assert_called_once()
                if log_wandb:
                    initialize.assert_called_once()
                    self.assertEqual(run.log.call_count, 3)
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
                self.assertEqual(len(csv_paths), 3 if log_csv else 0)
                for path in csv_paths:
                    with path.open() as file:
                        rows = list(csv.DictReader(file))
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["global_step"], "1")
                    if path.name == "eval.csv":
                        self.assertEqual(
                            rows[0],
                            {
                                "global_step": "1",
                                "mean_reward": "0.5",
                                "num_episodes": "1",
                                "video_paths": "[]",
                            },
                        )

                checkpoints = list(root.rglob("*.pt"))
                self.assertEqual(len(checkpoints), int(save_checkpoints))
                if save_checkpoints:
                    self.assertEqual(checkpoints[0].parent.name, "checkpoints")
                    load_policy(checkpoints[0])
                else:
                    self.assertEqual(list(root.rglob("checkpoints")), [])

                if log_wandb and save_checkpoints:
                    artifact.assert_called_once()
                    run.log_artifact.assert_called_once()
                    run.log_artifact.return_value.wait.assert_called_once()
                else:
                    artifact.assert_not_called()
                    run.log_artifact.assert_not_called()

    def test_disabled_logger_skips_video_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = train.ExperimentLogger(Path(directory), log_csv=False)
            logger.log_eval(
                global_step=1,
                mean_reward=0.5,
                num_episodes=1,
                video_paths=(Path(directory) / "missing.mp4",),
            )
            self.assertEqual(list(Path(directory).iterdir()), [])
