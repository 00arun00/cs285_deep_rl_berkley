"""Numerical contracts checked against small, independent analytical answers."""

from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch
from hw1_imitation import evaluation
from hw1_imitation.data import Normalizer
from hw1_imitation.model import PolicyConfig, build_policy
from hw1_imitation.randomness import RandomStreamFactory
from hypothesis import example, given
from hypothesis import strategies as st
from torch.utils.data import DataLoader, TensorDataset


def zero_policy(policy_type, *, state_dim=3, action_dim=1, horizon=2):
    policy = build_policy(
        PolicyConfig(policy_type, state_dim, action_dim, horizon, hidden_dims=()),
        cpu_generator=torch.Generator().manual_seed(42),
    )
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    return policy


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("training", [False, True], ids=["eval-mode", "train-mode"])
def test_validation_loss_weights_examples_not_batches(batch_size, training):
    policy = zero_policy("mse")
    policy.train(training)
    # The zero policy's squared errors sum to 250 over ten action elements:
    # per-example losses are [2, 5, 10, 26, 82], whose mean is exactly 25.
    targets = torch.tensor(
        [[0.0, 2.0], [1.0, 3.0], [2.0, 4.0], [4.0, 6.0], [8.0, 10.0]]
    ).unsqueeze(-1)
    loader = DataLoader(
        TensorDataset(torch.zeros(5, 3), targets), batch_size=batch_size, shuffle=False
    )
    result = evaluation.compute_validation_loss(
        policy,
        loader,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )
    assert result.examples_count == 5
    assert result.loss_mean == pytest.approx(25.0)
    assert policy.training == training


@pytest.mark.parametrize("training", [False, True], ids=["eval-mode", "train-mode"])
def test_empty_validation_has_no_numeric_mean(training):
    policy = zero_policy("mse")
    policy.train(training)
    loader = DataLoader(
        TensorDataset(torch.empty(0, 3), torch.empty(0, 2, 1)), batch_size=2
    )
    with pytest.raises(ValueError):
        evaluation.compute_validation_loss(
            policy,
            loader,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(7),
        )
    assert policy.training == training


class ScriptedRewardsEnv(gym.Env):
    """Only the external environment is substituted; rollout scoring stays real."""

    def __init__(self, rewards, end_signal):
        self.rewards = rewards
        self.end_signal = end_signal
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.episode = -1
        self.position = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.position = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        reward = self.rewards[self.episode][self.position]
        self.position += 1
        done = self.position == len(self.rewards[self.episode])
        return (
            np.zeros(3, dtype=np.float32),
            float(reward),
            done and self.end_signal == "terminated",
            done and self.end_signal == "truncated",
            {},
        )

    def close(self):
        self.closed = True


@pytest.mark.parametrize("end_signal", ["terminated", "truncated"])
@given(
    rewards=st.lists(
        st.lists(st.integers(-10, 10), min_size=1, max_size=6), min_size=1, max_size=4
    )
)
@example(rewards=[[0, 8, 2], [4], [1, 3, 9, 2]])
@example(rewards=[[-4, -1, -3], [-7]])
@example(rewards=[[0]])
def test_evaluation_score_is_mean_of_episode_maxima(end_signal, rewards):
    env = ScriptedRewardsEnv(rewards, end_signal)
    policy = zero_policy("mse")
    normalizer = Normalizer(
        np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    with patch.object(evaluation.gym, "make", return_value=env):
        result = evaluation.evaluate_policy(
            policy,
            normalizer,
            torch.device("cpu"),
            chunk_size=2,
            video_size=(16, 16),
            num_video_episodes=0,
            flow_num_steps=1,
            streams=RandomStreamFactory(42),
            num_eval_episodes=len(rewards),
        )
    # Episode length cannot change its weight, and its final reward need not be best.
    assert result.mean_reward == pytest.approx(
        sum(max(episode) for episode in rewards) / len(rewards)
    )
    assert result.num_episodes == len(rewards)
    assert env.episode + 1 == len(rewards)
    assert env.closed


@pytest.mark.parametrize("batch_size,horizon,action_dim", [(1, 1, 1), (3, 2, 2)])
@pytest.mark.parametrize("num_steps", [1, 2, 7, 16])
@pytest.mark.parametrize("field", ["zero", "constant"])
def test_flow_integration_matches_constant_velocity_solution(
    batch_size, horizon, action_dim, num_steps, field
):
    policy = zero_policy("flow", horizon=horizon, action_dim=action_dim)
    # With no hidden layers and all weights zero, the sole bias is the constant
    # output velocity. Configure real parameters rather than replacing _forward.
    velocity = torch.zeros(horizon * action_dim)
    if field == "constant":
        velocity = torch.arange(horizon * action_dim, dtype=torch.float32) * 0.75 - 0.5
    with torch.no_grad():
        for parameter in policy.parameters():
            if parameter.ndim == 1:
                parameter.copy_(velocity)

    generator = torch.Generator().manual_seed(123)
    reference_generator = torch.Generator().manual_seed(123)
    initial = torch.randn(
        batch_size, horizon * action_dim, generator=reference_generator
    )
    states = torch.arange(batch_size * 3, dtype=torch.float32).reshape(batch_size, 3)
    actual = policy.sample_actions(states, generator=generator, num_steps=num_steps)
    # dx/dt = v over [0, 1] has x(1) = x(0) + v for every positive step count.
    expected = (initial + velocity).reshape(batch_size, horizon, action_dim)
    # Repeated float32 additions need tolerance; no cross-device equality assumed.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
