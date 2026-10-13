"""Plateau boundaries and graceful training shutdown."""

from unittest.mock import patch

import pytest
import torch

from hw1_imitation import train
from hw1_imitation.early_stopping import EarlyStopping
from tests import test_training_reporting

run_training = test_training_reporting.run_training


@pytest.mark.parametrize("objective, direction", [("max", 1), ("min", -1)])
def test_warmup_reference_persists_and_patience_counts_evaluations(
    objective, direction
):
    stopper = EarlyStopping(
        patience=2, min_delta=0.125, start_step=10, objective=objective
    )
    assert not stopper.update(step=1, value=direction * 0.5)
    assert not stopper.update(step=5, value=direction * 1.0)
    assert not stopper.update(step=9, value=direction * 0.5)
    assert not stopper.update(step=10, value=direction * 0.5)
    # These improve on each other, but not on the reference from warm-up.
    # Spaced-out steps ensure patience counts evaluations, not optimizer steps.
    assert not stopper.update(step=20, value=direction * 0.625)
    assert stopper.update(step=30, value=direction * 0.75)


@pytest.mark.parametrize("objective, direction", [("max", 1), ("min", -1)])
def test_small_gains_accumulate_and_reset_patience(objective, direction):
    stopper = EarlyStopping(
        patience=3, min_delta=0.125, start_step=0, objective=objective
    )
    # Binary fractions make the exact min_delta boundary unambiguous.
    assert not stopper.update(step=1, value=direction * 0.5)
    assert not stopper.update(step=2, value=direction * 0.5625)
    assert not stopper.update(step=3, value=direction * 0.625)
    assert not stopper.update(step=4, value=direction * 0.6875)
    assert not stopper.update(step=5, value=direction * 0.6875)
    assert not stopper.update(step=6, value=direction * 0.6875)
    assert stopper.update(step=7, value=direction * 0.6875)


@pytest.mark.parametrize("objective, direction", [("max", 1), ("min", -1)])
def test_zero_delta_still_requires_strict_improvement(objective, direction):
    stopper = EarlyStopping(patience=1, min_delta=0, start_step=0, objective=objective)
    assert not stopper.update(step=1, value=direction * 0.5)
    assert not stopper.update(step=2, value=direction * 0.6)
    assert stopper.update(step=3, value=direction * 0.6)


@pytest.mark.parametrize("objective", ["min", "max"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_value_rejected_even_during_warmup(objective, value):
    with pytest.raises(ValueError):
        EarlyStopping(1, 0, 100, objective=objective).update(step=1, value=value)


def test_invalid_objective_rejected():
    with pytest.raises(ValueError):
        EarlyStopping(patience=1, min_delta=0, start_step=0, objective="invalid")


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
        save_checkpoints=False,
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
