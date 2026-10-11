"""Explicit output roots preserve local run naming and collision protection."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hw1_imitation import train
from hw1_imitation.data import Episode, EpisodesDataset
from hw1_imitation.evaluation import EvaluationResults


@pytest.mark.parametrize("output_root", [None, Path("custom/experiments")])
def test_training_outputs_use_selected_root_and_reject_collisions(
    output_root, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    episodes = EpisodesDataset(
        tuple(
            Episode(
                index,
                np.zeros((2, 3), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
            )
            for index in range(5)
        )
    )
    config = train.TrainConfig(
        hidden_dims=(4,),
        chunk_size=1,
        num_epochs=1,
        batch_size=8,
        final_eval_episodes=1,
        num_video_episodes=0,
        log_wandb=False,
        show_summary=False,
        exp_name="outputs",
    )
    with (
        patch.object(train, "datetime") as clock,
        patch.object(train, "download_pusht", return_value=tmp_path),
        patch.object(train, "load_episodes_dataset", return_value=episodes),
        patch.object(train.torch.accelerator, "current_accelerator", return_value=None),
        patch.object(
            train, "evaluate_policy", return_value=EvaluationResults(0.5, 1, ())
        ),
        patch.object(train, "train_step", wraps=train.train_step) as step,
    ):
        clock.now.return_value = datetime(2026, 10, 11, 12, 34, 56)
        train.run_training(config, output_root=output_root)
        expected_root = Path("exp") if output_root is None else output_root
        run_dir = expected_root / "seed_42_20261011_123456_outputs"
        assert sorted(
            path.relative_to(run_dir).as_posix()
            for path in run_dir.rglob("*")
            if path.is_file()
        ) == [
            "checkpoints/policy_step_1.pt",
            "eval.csv",
            "train.csv",
            "validation.csv",
        ]
        assert list(expected_root.iterdir()) == [run_dir]
        if output_root is not None:
            assert not Path("exp").exists()

        checkpoint = run_dir / "checkpoints/policy_step_1.pt"
        original_bytes = checkpoint.read_bytes()
        with pytest.raises(FileExistsError):
            train.run_training(config, output_root=output_root)
        assert checkpoint.read_bytes() == original_bytes
        assert step.call_count == 1
    assert Path.cwd() == tmp_path
