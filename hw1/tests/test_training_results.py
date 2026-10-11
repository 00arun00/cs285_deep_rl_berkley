"""Completed training results report real budgets and retained local files."""

from hw1_imitation.config import (
    CheckpointConfig,
    EvaluationConfig,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
)

from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import numpy as np
import pytest

from hw1_imitation import train
from hw1_imitation.data import Episode, EpisodesDataset
from hw1_imitation.evaluation import EvaluationResults


@pytest.fixture
def run_training(tmp_path, monkeypatch):
    episodes = EpisodesDataset(
        tuple(
            Episode(
                index,
                np.zeros((6, 3), dtype=np.float32),
                np.zeros((6, 2), dtype=np.float32),
            )
            for index in range(5)
        )
    )
    monkeypatch.setattr(train, "download_pusht", lambda _: tmp_path)
    monkeypatch.setattr(train, "load_episodes_dataset", lambda **_: episodes)
    monkeypatch.setattr(train.torch.accelerator, "current_accelerator", lambda: None)
    defaults = train.TrainConfig(
        model=ModelConfig(hidden_dims=(4,), chunk_size=1),
        optimization=OptimizationConfig(num_epochs=2, batch_size=5),
        logging=LoggingConfig(interval=3, wandb=False, csv=False, show_summary=False),
        evaluation=EvaluationConfig(
            interval=3, episodes=2, final_episodes=5, num_video_episodes=0
        ),
        checkpoints=CheckpointConfig(top_k=1),
    )

    def run(**changes):
        evaluations = []
        scores = iter((0.9, 0.9, 0.1))

        def evaluate(**kwargs):
            score = next(scores)
            count = kwargs["num_eval_episodes"]
            result = EvaluationResults(score, count, (), (score,) * count)
            evaluations.append(result)
            return result

        monkeypatch.setattr(train, "evaluate_policy", evaluate)
        output_dir = tmp_path / "run"
        output_dir.mkdir()
        result = train.run_training(replace(defaults, **changes), output_dir=output_dir)
        return result, evaluations

    return run


@pytest.mark.parametrize(
    "saving,top_k,retained_steps,best_step",
    [(True, 1, {3, 8}, 3), (True, 0, {8}, None), (False, 3, set(), None)],
)
def test_result_counts_actual_examples_and_references_retained_checkpoints(
    run_training, saving, top_k, retained_steps, best_step
):
    result, evaluations = run_training(
        checkpoints=CheckpointConfig(enabled=saving, top_k=top_k)
    )

    # Four training episodes provide 24 samples/epoch. Each epoch drops four
    # examples; the counter must also survive the separate logging-window resets.
    assert result.optimizer_steps == 8
    assert result.examples_processed == 40
    assert result.final_evaluation is evaluations[-1]
    assert [evaluation.num_episodes for evaluation in evaluations] == [2, 2, 5]
    assert result.final_evaluation.episode_max_rewards == (0.1,) * 5
    assert result.pending_cleanup_steps == ()
    assert result.output_dir.is_dir()
    checkpoint_dir = result.output_dir / "checkpoints"
    expected_files = {
        checkpoint_dir / f"policy_step_{step}.pt" for step in retained_steps
    }
    assert set(checkpoint_dir.glob("*.pt")) == expected_files
    assert result.latest_checkpoint == (
        checkpoint_dir / "policy_step_8.pt" if saving else None
    )
    assert result.best_checkpoint == (
        checkpoint_dir / f"policy_step_{best_step}.pt"
        if best_step is not None
        else None
    )
    with pytest.raises(FrozenInstanceError):
        result.optimizer_steps = 0


@pytest.mark.parametrize("top_k,pending", [(0, (3, 6)), (1, (6,))])
def test_result_excludes_pending_remote_deletions_after_tracking_closes(
    run_training, monkeypatch, top_k, pending
):
    tracking_closed = []
    run = Mock(id="training-result", offline=False)
    run.log_artifact.side_effect = lambda artifact, **_: artifact

    @contextmanager
    def tracking(**kwargs):
        yield run
        tracking_closed.append(True)

    def artifact(**kwargs):
        result = Mock()
        result.delete.side_effect = RuntimeError("remote deletion unavailable")
        return result

    monkeypatch.setattr(train.wandb, "init", tracking)
    monkeypatch.setattr(train.wandb, "Artifact", artifact)
    result, _ = run_training(
        logging=LoggingConfig(wandb=True, csv=False, show_summary=False),
        checkpoints=CheckpointConfig(top_k=top_k),
    )

    assert tracking_closed == [True]
    assert result.pending_cleanup_steps == pending
    assert result.latest_checkpoint.is_file()
    assert result.latest_checkpoint.name == "policy_step_8.pt"
    if top_k:
        assert result.best_checkpoint.is_file()
        assert result.best_checkpoint.name == "policy_step_3.pt"
    else:
        assert result.best_checkpoint is None
    for step in pending:
        assert not (
            result.output_dir / "checkpoints" / f"policy_step_{step}.pt"
        ).exists()


def test_tracking_cleanup_failure_prevents_successful_return(run_training, monkeypatch):
    @contextmanager
    def tracking(**kwargs):
        yield Mock(offline=False)
        raise RuntimeError("tracking cleanup failed")

    monkeypatch.setattr(train.wandb, "init", tracking)
    with pytest.raises(RuntimeError, match="tracking cleanup failed"):
        run_training(
            logging=LoggingConfig(wandb=True, csv=False, show_summary=False),
            checkpoints=CheckpointConfig(enabled=False),
        )
