"""Evaluation callbacks observe completed work without owning training state."""

from hw1_imitation.config import (
    CheckpointConfig,
    EvaluationConfig,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
)

import csv
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from hw1_imitation import evaluation, train
from hw1_imitation.data import Episode, EpisodesDataset


class RecordingEnv(gym.Env):
    """Small seeded rollouts with observable actions and resource ownership."""

    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.seeds = []
        self.actions = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seeds.append(seed)
        self.steps = 0
        return self.np_random.normal(size=3).astype(np.float32), {}

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        observation = self.np_random.normal(size=3).astype(np.float32)
        return observation, self.steps / 2, self.steps == 2, False, {}

    def close(self):
        self.closed = True


@pytest.fixture
def training_case(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    episodes = EpisodesDataset(
        tuple(
            Episode(
                index,
                rng.normal(size=(6, 3)).astype(np.float32),
                rng.normal(size=(6, 2)).astype(np.float32),
            )
            for index in range(5)
        )
    )
    environments = []

    def make_env(*args, **kwargs):
        environment = RecordingEnv()
        environments.append(environment)
        return environment

    monkeypatch.setattr(train, "download_pusht", lambda _: tmp_path)
    monkeypatch.setattr(train, "load_episodes_dataset", lambda **_: episodes)
    monkeypatch.setattr(train.torch.accelerator, "current_accelerator", lambda: None)
    monkeypatch.setattr(evaluation.gym, "make", make_env)
    config = train.TrainConfig(
        model=ModelConfig(policy_type="flow", hidden_dims=(4,), chunk_size=1),
        optimization=OptimizationConfig(num_epochs=2, batch_size=5),
        logging=LoggingConfig(interval=3, wandb=False, show_summary=False),
        evaluation=EvaluationConfig(
            validation_interval=3,
            interval=3,
            episodes=2,
            final_episodes=5,
            flow_num_steps=2,
            num_video_episodes=0,
        ),
        checkpoints=CheckpointConfig(top_k=0),
    )
    return config, environments


@pytest.mark.parametrize("saving", [False, True])
@pytest.mark.parametrize(
    "interval,expected",
    [
        (3, [(3, 15, False), (6, 30, False), (8, 40, True)]),
        (4, [(4, 20, False), (8, 40, True)]),
    ],
)
def test_events_follow_logging_saving_and_actual_training_budgets(
    training_case, tmp_path, saving, interval, expected
):
    config, environments = training_case
    events = []

    def observe(event):
        assert all(environment.closed for environment in environments)
        with (tmp_path / "eval.csv").open() as file:
            rows = list(csv.DictReader(file))
        assert int(rows[-1]["global_step"]) == event.optimizer_steps
        assert float(rows[-1]["mean_reward"]) == event.evaluation.mean_reward
        if saving:
            assert event.checkpoint.is_file()
            assert event.checkpoint.name == f"policy_step_{event.optimizer_steps}.pt"
        else:
            assert event.checkpoint is None
        events.append(event)

    result = train.run_training(
        replace(
            config,
            checkpoints=replace(config.checkpoints, enabled=saving),
            evaluation=replace(config.evaluation, interval=interval),
        ),
        output_dir=tmp_path,
        on_evaluation=observe,
    )
    assert [
        (event.optimizer_steps, event.examples_processed, event.is_final)
        for event in events
    ] == expected
    assert [event.evaluation.num_episodes for event in events] == [2] * (
        len(events) - 1
    ) + [5]
    assert events[-1].evaluation is result.final_evaluation
    assert events[-1].checkpoint == result.latest_checkpoint
    if saving:
        assert not events[0].checkpoint.exists()  # Later retention may remove it.
    with pytest.raises(FrozenInstanceError):
        events[-1].optimizer_steps = 0


def test_noop_callback_preserves_metrics_checkpoints_rollouts_and_seeds(
    training_case, tmp_path
):
    config, environments = training_case
    (tmp_path / "baseline").mkdir()
    (tmp_path / "observed").mkdir()
    baseline = train.run_training(config, output_dir=tmp_path / "baseline")
    baseline_envs = list(environments)
    environments.clear()
    observed = train.run_training(
        config, output_dir=tmp_path / "observed", on_evaluation=lambda event: None
    )

    assert baseline.final_evaluation == observed.final_evaluation
    for name in ("train.csv", "validation.csv", "eval.csv"):
        assert (baseline.output_dir / name).read_text() == (
            observed.output_dir / name
        ).read_text()
    expected_files = sorted((baseline.output_dir / "checkpoints").glob("*.pt"))
    actual_files = sorted((observed.output_dir / "checkpoints").glob("*.pt"))
    assert [path.name for path in expected_files] == [
        path.name for path in actual_files
    ]
    for expected, actual in zip(expected_files, actual_files, strict=True):
        expected_payload = torch.load(expected, weights_only=True)
        actual_payload = torch.load(actual, weights_only=True)
        assert actual_payload.pop("model_config") == expected_payload.pop(
            "model_config"
        )
        torch.testing.assert_close(
            actual_payload,
            expected_payload,
            rtol=0,
            atol=0,
        )
    for expected, actual in zip(baseline_envs, environments, strict=True):
        assert actual.seeds == expected.seeds
        np.testing.assert_array_equal(actual.actions, expected.actions)
        assert actual.closed and expected.closed


@pytest.mark.parametrize("failure", ["evaluation", "logging", "saving"])
def test_failed_evaluation_outputs_do_not_emit_events(
    training_case, tmp_path, monkeypatch, failure
):
    config, environments = training_case
    owner, attribute = {
        "evaluation": (train, "evaluate_policy"),
        "logging": (train.ExperimentLogger, "log_eval"),
        "saving": (train, "save_checkpoint_and_retain"),
    }[failure]
    monkeypatch.setattr(owner, attribute, Mock(side_effect=RuntimeError(failure)))
    callback = Mock()
    with pytest.raises(RuntimeError, match=failure):
        train.run_training(config, output_dir=tmp_path, on_evaluation=callback)
    callback.assert_not_called()
    assert all(environment.closed for environment in environments)


@pytest.mark.parametrize("failure", ["callback", "cleanup"])
def test_callback_and_cleanup_exceptions_propagate_after_contexts_close(
    training_case, tmp_path, monkeypatch, failure
):
    config, environments = training_case
    closed = []
    events = []
    real_progress = train.tqdm

    @contextmanager
    def progress(*args, **kwargs):
        try:
            with real_progress(*args, **kwargs) as bar:
                yield bar
        finally:
            closed.append("progress")

    @contextmanager
    def tracking(**kwargs):
        try:
            yield Mock(offline=False)
        finally:
            closed.append("tracking")
            if failure == "cleanup":
                raise RuntimeError("cleanup failed")

    def observe(event):
        assert all(environment.closed for environment in environments)
        events.append(event)
        if failure == "callback":
            raise RuntimeError("callback failed")

    step = Mock(wraps=train.train_step)
    monkeypatch.setattr(train, "train_step", step)
    monkeypatch.setattr(train, "tqdm", progress)
    monkeypatch.setattr(train.wandb, "init", tracking)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        train.run_training(
            replace(
                config,
                logging=replace(config.logging, wandb=True),
                checkpoints=replace(config.checkpoints, enabled=False),
            ),
            output_dir=tmp_path,
            on_evaluation=observe,
        )
    assert step.call_count == (3 if failure == "callback" else 8)
    assert events[-1].is_final == (failure == "cleanup")
    assert closed == ["progress", "tracking"]
