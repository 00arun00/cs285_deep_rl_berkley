"""Plateau boundaries and graceful training shutdown."""

from unittest.mock import patch

import pytest
import torch

from hw1_imitation import train
from hw1_imitation.early_stopping import EarlyStopping
from tests import test_training_reporting

run_training = test_training_reporting.run_training


def test_warmup_baseline_and_exact_patience():
    stopper = EarlyStopping(patience=2, min_delta=0.1, start_step=10)
    assert not stopper.update(step=9, reward=1.0)
    assert not stopper.update(step=10, reward=0.5)
    assert not stopper.update(step=11, reward=0.5)
    assert stopper.update(step=12, reward=0.5)


def test_small_gains_accumulate_and_reset_patience():
    stopper = EarlyStopping(patience=3, min_delta=0.125, start_step=0)
    assert not stopper.update(step=1, reward=0.5)
    assert not stopper.update(step=2, reward=0.5625)
    assert not stopper.update(step=3, reward=0.625)
    assert not stopper.update(step=4, reward=0.625)
    assert not stopper.update(step=5, reward=0.625)
    assert stopper.update(step=6, reward=0.625)


def test_zero_delta_still_requires_strict_improvement():
    stopper = EarlyStopping(patience=1, min_delta=0, start_step=0)
    assert not stopper.update(step=1, reward=0.5)
    assert not stopper.update(step=2, reward=0.6)
    assert stopper.update(step=3, reward=0.6)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_reward_rejected_even_during_warmup(reward):
    with pytest.raises(ValueError, match="finite"):
        EarlyStopping(1, 0, 100).update(step=1, reward=reward)


@pytest.mark.parametrize(
    "settings",
    [
        {"early_stopping_patience": 0},
        {"early_stopping_min_epochs": -1},
        {"early_stopping_min_delta": -0.1},
        {"early_stopping_min_delta": float("nan")},
    ],
)
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        train.TrainConfig(**settings)


def test_stopping_exits_both_loops_and_flushes_loss(run_training):
    # The fixture has 24 training examples: one step per epoch here.
    cfg = train.TrainConfig(
        hidden_dims=(4,),
        chunk_size=1,
        num_epochs=10,
        batch_size=24,
        eval_interval=1,
        log_interval=100,
        num_video_episodes=0,
        log_wandb=False,
        log_csv=False,
        show_summary=False,
        early_stopping_patience=2,
        early_stopping_min_epochs=1,
    )
    with (
        patch.object(
            train,
            "train_step",
            side_effect=[torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)],
        ) as step,
        patch.object(train, "ExperimentLogger") as logger,
    ):
        run_training(cfg)
    assert step.call_count == 3
    logger.return_value.log_train.assert_called_once_with(
        global_step=3,
        epoch=2,
        loss_window_mean=2.0,
        window_examples_count=72,
    )
    assert logger.return_value.log_eval.call_count == 3
