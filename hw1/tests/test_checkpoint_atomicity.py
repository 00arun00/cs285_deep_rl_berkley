"""A failed serialization must not publish a partial policy checkpoint."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hw1_imitation.checkpoint import load_policy, save_policy
from hw1_imitation.data import Normalizer
from hw1_imitation.model import PolicyConfig, build_policy


def policy_and_stats(seed, offset):
    policy = build_policy(
        PolicyConfig("mse", state_dim=3, action_dim=2, chunk_size=2, hidden_dims=(4,)),
        cpu_generator=torch.Generator().manual_seed(seed),
    )
    normalizer = Normalizer(
        np.full(3, offset, dtype=np.float32),
        np.full(3, offset + 1, dtype=np.float32),
        np.full(2, offset, dtype=np.float32),
        np.full(2, offset + 1, dtype=np.float32),
    )
    return policy, normalizer


def assert_saved_policy(path, expected_policy, expected_stats, flow_num_steps):
    restored, stats, inference = load_policy(path)
    assert restored.config == expected_policy.config
    for name, expected in expected_policy.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name], expected, rtol=0, atol=0
        )
    for name, expected in expected_stats.state_dict().items():
        torch.testing.assert_close(stats.state_dict()[name], expected, rtol=0, atol=0)
    assert inference["flow_num_steps"] == flow_num_steps


def failing_serializer(partial_write):
    def serialize(payload, destination, *args, **kwargs):
        if partial_write:
            # torch.save supports paths and binary file objects. Inject the fault
            # at that library boundary without assuming a temporary-file scheme.
            if hasattr(destination, "write"):
                destination.write(b"incomplete checkpoint")
                destination.flush()
            else:
                Path(destination).write_bytes(b"incomplete checkpoint")
        raise OSError("injected serialization failure")

    return serialize


@pytest.mark.parametrize(
    "partial_write", [False, True], ids=["before-write", "partial-write"]
)
def test_serialization_failure_preserves_existing_checkpoint(tmp_path, partial_write):
    path = tmp_path / "policy.pt"
    original, original_stats = policy_and_stats(42, 0)
    replacement, replacement_stats = policy_and_stats(43, 2)
    save_policy(path, original, original_stats, flow_num_steps=3)
    original_bytes = path.read_bytes()

    with patch.object(torch, "save", side_effect=failing_serializer(partial_write)):
        with pytest.raises(OSError, match="injected serialization failure"):
            save_policy(path, replacement, replacement_stats, flow_num_steps=7)

    assert path.read_bytes() == original_bytes
    assert_saved_policy(path, original, original_stats, flow_num_steps=3)

    # A later successful save must replace the old policy, not silently retain it.
    save_policy(path, replacement, replacement_stats, flow_num_steps=7)
    assert_saved_policy(path, replacement, replacement_stats, flow_num_steps=7)


@pytest.mark.parametrize(
    "partial_write", [False, True], ids=["before-write", "partial-write"]
)
def test_serialization_failure_does_not_publish_new_checkpoint(tmp_path, partial_write):
    path = tmp_path / "policy.pt"
    policy, stats = policy_and_stats(42, 0)
    with patch.object(torch, "save", side_effect=failing_serializer(partial_write)):
        with pytest.raises(OSError, match="injected serialization failure"):
            save_policy(path, policy, stats, flow_num_steps=3)

    assert not path.exists()
    save_policy(path, policy, stats, flow_num_steps=3)
    assert_saved_policy(path, policy, stats, flow_num_steps=3)
