"""Training reports actual loss windows and unresolved checkpoint cleanup."""

from hw1_imitation.config import (
    CheckpointConfig,
    EvaluationConfig,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
)

import logging
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hw1_imitation import train
from hw1_imitation.data import Episode, EpisodesDataset
from hw1_imitation.evaluation import EvaluationResults


@pytest.fixture
def run_training(tmp_path):
    episodes = EpisodesDataset(
        tuple(
            Episode(
                i,
                np.zeros((6, 3), dtype=np.float32),
                np.zeros((6, 2), dtype=np.float32),
            )
            for i in range(5)
        )
    )

    def run(config):
        with (
            patch.object(train, "download_pusht", return_value=tmp_path),
            patch.object(train, "load_episodes_dataset", return_value=episodes),
            patch.object(
                train.torch.accelerator, "current_accelerator", return_value=None
            ),
            patch.object(
                train,
                "evaluate_policy",
                return_value=EvaluationResults(0.5, 1, (), (0.5,)),
            ),
        ):
            output_dir = tmp_path / "run"
            output_dir.mkdir(parents=True)
            train.run_training(config, output_dir=output_dir)

    return run


def config(*, num_epochs, batch_size, log_interval=100, save_checkpoints=True):
    return train.TrainConfig(
        model=ModelConfig(hidden_dims=(4,), chunk_size=1),
        evaluation=EvaluationConfig(
            num_video_episodes=0, validation_interval=100, interval=100
        ),
        logging=LoggingConfig(
            wandb=False, csv=False, show_summary=False, interval=log_interval
        ),
        optimization=OptimizationConfig(num_epochs=num_epochs, batch_size=batch_size),
        checkpoints=CheckpointConfig(enabled=save_checkpoints),
    )


@pytest.mark.parametrize(
    "epochs,batch_size,interval,expected",
    [
        (1, 4, 4, [(4, 16, 2.5), (6, 8, 5.5)]),
        (2, 3, 5, [(5, 15, 3.0), (10, 15, 8.0), (15, 15, 13.0), (16, 3, 16.0)]),
        (1, 4, 100, [(6, 24, 3.5)]),
    ],
)
def test_training_logs_actual_loss_windows(
    run_training, epochs, batch_size, interval, expected
):
    """Training reports complete loss windows across epoch and run boundaries.

    Protects:
        Means and example counts describe the actual window, including a short final
        window.
    Value:
        Catches misleading learning curves from dropped, reset, or incorrectly sized
        windows.
    Approach:
        Feed known step losses into the real driver and inspect public logger
        records.
    """
    # Known per-step losses make the aggregation oracle independent of the model.
    losses = [torch.tensor(float(i)) for i in range(1, epochs * (24 // batch_size) + 1)]
    with (
        patch.object(train, "train_step", side_effect=losses),
        patch.object(train, "ExperimentLogger") as logger,
    ):
        run_training(
            config(
                num_epochs=epochs,
                batch_size=batch_size,
                log_interval=interval,
                save_checkpoints=False,
            )
        )
    records = [call.kwargs for call in logger.return_value.log_train.call_args_list]
    assert [
        (r["global_step"], r["window_examples_count"], r["loss_window_mean"])
        for r in records
    ] == expected


@pytest.mark.parametrize("pending", [set(), {1, 7}], ids=["clean", "pending"])
def test_training_warns_about_pending_checkpoint_cleanup(run_training, caplog, pending):
    """Makes leftover checkpoints visible at shutdown without warning after successful
    cleanup."""
    with (
        patch.object(train, "save_checkpoint_and_retain", return_value=pending),
        caplog.at_level(logging.WARNING),
    ):
        run_training(config(num_epochs=1, batch_size=24, save_checkpoints=True))
    warnings = [
        r.getMessage().lower()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "cleanup" in r.getMessage().lower()
    ]
    if pending:
        assert len(warnings) == 1
        assert "pending" in warnings[0]
        assert all(str(step) in warnings[0] for step in pending)
    else:
        assert warnings == []
