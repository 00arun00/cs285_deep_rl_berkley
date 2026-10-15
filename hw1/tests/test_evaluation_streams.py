"""Evaluation stream selection preserves replay and isolates final rollouts."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from hw1_imitation import evaluation
from hw1_imitation.randomness import RandomStreamFactory, StreamId
from tests.test_randomness import TinyEnv, make_normalizer, make_policy


def rollout(model, **stream_ids):
    env = TinyEnv()
    with patch.object(evaluation.gym, "make", return_value=env):
        evaluation.evaluate_policy(
            model,
            make_normalizer(),
            torch.device("cpu"),
            chunk_size=1,
            video_size=(16, 16),
            num_video_episodes=0,
            flow_num_steps=2,
            streams=RandomStreamFactory(42),
            num_eval_episodes=2,
            **stream_ids,
        )
    return np.array(env.initial_observations), np.array(env.actions)


def test_default_streams_preserve_periodic_rollouts():
    model = make_policy()
    default = rollout(model)
    explicit = rollout(
        model, env_stream=StreamId.EVAL_ENV, policy_stream=StreamId.EVAL_POLICY
    )
    for actual, expected in zip(default, explicit, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "change_env,change_policy", [(True, False), (False, True), (True, True)]
)
def test_final_streams_are_separate_and_replayable(change_env, change_policy):
    model = make_policy()
    periodic_obs, periodic_actions = rollout(model)
    stream_ids = {
        "env_stream": StreamId.FINAL_EVAL_ENV if change_env else StreamId.EVAL_ENV,
        "policy_stream": StreamId.FINAL_EVAL_POLICY
        if change_policy
        else StreamId.EVAL_POLICY,
    }
    final_obs, final_actions = rollout(model, **stream_ids)
    replay_obs, replay_actions = rollout(model, **stream_ids)
    np.testing.assert_array_equal(final_obs, replay_obs)
    np.testing.assert_array_equal(final_actions, replay_actions)
    if change_env:
        assert not np.array_equal(periodic_obs, final_obs)
    else:
        np.testing.assert_array_equal(periodic_obs, final_obs)
    if change_policy:
        assert not np.array_equal(periodic_actions, final_actions)
