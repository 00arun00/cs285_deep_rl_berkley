"""Retention over generated histories, including cleanup failure and recovery."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hw1_imitation import evaluation
from hw1_imitation.data import Normalizer
from hw1_imitation.model import PolicyConfig, build_policy
from hypothesis import example, given, settings
from hypothesis import strategies as st


def expected_retention(history, top_k, latest):
    """Rank by counting better predecessors, independently of production sorting."""
    best = {
        step
        for step, score in history.items()
        if sum(
            other_score > score or (other_score == score and other_step < step)
            for other_step, other_score in history.items()
        )
        < top_k
    }
    return best | {latest}


class ArtifactService:
    """A minimal external artifact service with controllable deletion failures."""

    offline = False
    id = "retention-property"

    def __init__(self):
        self.live_steps = set()
        self.aliases = {}
        self.fail_delete = False

    def artifact(self, *, name, type, metadata):
        service = self
        step = metadata["step"]

        class Artifact:
            def add_file(self, path, *, name):
                assert Path(path).is_file()

            def wait(self):
                service.live_steps.add(step)
                for alias in self.aliases:
                    service.aliases[alias] = step

            def delete(self):
                # Deleting a still-aliased artifact is never safe.
                assert step not in service.aliases.values()
                if service.fail_delete:
                    raise RuntimeError("injected remote cleanup failure")
                service.live_steps.remove(step)

        return Artifact()

    def log_artifact(self, artifact, *, aliases):
        artifact.aliases = aliases
        return artifact


def make_policy_and_stats():
    policy = build_policy(
        PolicyConfig("mse", state_dim=2, action_dim=1, chunk_size=1, hidden_dims=()),
        cpu_generator=torch.Generator().manual_seed(42),
    )
    stats = Normalizer(
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    return policy, stats


def save(policy, stats, root, records, step, score, top_k, run):
    return evaluation.save_checkpoint_and_retain(
        policy,
        step,
        normalizer=stats,
        flow_num_steps=1,
        mean_reward=score,
        top_k=top_k,
        checkpoints=records,
        checkpoint_dir=root,
        run=run,
    )


def assert_aliases(service, history, top_k, latest):
    expected = {"latest": latest}
    if top_k:
        # max keeps the first (earliest) item on a tie in insertion-ordered history.
        expected["best"] = max(history, key=history.get)
    assert service.aliases == expected


@pytest.mark.parametrize("remote", [False, True], ids=["local-only", "remote"])
@settings(max_examples=50, deadline=None)
@given(
    scores=st.lists(st.integers(-5, 5), min_size=1, max_size=10),
    top_k=st.integers(0, 12),
    step_gap=st.integers(1, 7),
)
@example(scores=[3, 3, 3, 3], top_k=2, step_gap=7)
@example(scores=[5, 4, 3, 2, 1], top_k=0, step_gap=1)
@example(scores=[-2, -1, 0, 1], top_k=8, step_gap=3)
def test_retention_matches_full_history_after_every_save(
    remote, scores, top_k, step_gap
):
    policy, stats = make_policy_and_stats()
    service = ArtifactService()
    records, history, paths = {}, {}, {}
    # Hypothesis examples need fresh resources; pytest fixtures reset only per test.
    with (
        TemporaryDirectory() as directory,
        patch.object(
            evaluation.wandb, "Artifact", side_effect=service.artifact
        ) as artifact_factory,
    ):
        for index, score in enumerate(scores, 1):
            step = index * step_gap
            history[step] = score
            pending = save(
                policy,
                stats,
                Path(directory),
                records,
                step,
                score,
                top_k,
                service if remote else None,
            )
            paths[step] = records[step].path
            keep = expected_retention(history, top_k, step)
            assert pending == set()
            assert set(records) == keep
            assert {saved for saved, path in paths.items() if path.is_file()} == keep
            assert {saved: record.score for saved, record in records.items()} == {
                saved: history[saved] for saved in keep
            }
            if remote:
                assert service.live_steps == keep
                assert_aliases(service, history, top_k, step)
        if not remote:
            artifact_factory.assert_not_called()


@pytest.mark.parametrize("failure_target", ["local", "remote"])
@settings(max_examples=50, deadline=None)
@given(
    events=st.lists(
        st.tuples(st.integers(-5, 5), st.booleans()), min_size=2, max_size=8
    ),
    top_k=st.integers(0, 4),
)
@example(events=[(3, False), (2, True), (1, True)], top_k=0)
@example(events=[(3, False), (3, False), (2, True), (1, True), (4, False)], top_k=2)
def test_cleanup_failures_preserve_retry_state_until_recovery(
    failure_target, events, top_k
):
    policy, stats = make_policy_and_stats()
    service = ArtifactService()
    records, history, paths = {}, {}, {}
    expected_tracked, expected_local, expected_remote = set(), set(), set()
    real_unlink = Path.unlink
    with (
        TemporaryDirectory() as directory,
        patch.object(evaluation.wandb, "Artifact", side_effect=service.artifact),
    ):
        # Always finish with a healthy save: all obsolete retry state must clear.
        for step, (score, fail_cleanup) in enumerate([*events, (0, False)], 1):
            history[step] = score
            keep = expected_retention(history, top_k, step)
            obsolete = (expected_tracked | {step}) - keep
            existing_paths = set(paths.values())
            service.fail_delete = fail_cleanup and failure_target == "remote"

            def unlink(path, *args, **kwargs):
                # Fault only old checkpoint files, never new serialization files.
                if (
                    fail_cleanup
                    and failure_target == "local"
                    and path in existing_paths
                ):
                    raise OSError("injected local cleanup failure")
                return real_unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", unlink):
                pending = save(
                    policy, stats, Path(directory), records, step, score, top_k, service
                )
            paths[step] = records[step].path

            if fail_cleanup:
                expected_tracked |= {step}
                expected_remote |= {step}
                expected_local = (
                    expected_local | {step} if failure_target == "local" else keep
                )
                assert pending == obsolete
            else:
                expected_tracked = keep.copy()
                expected_local = keep.copy()
                expected_remote = keep.copy()
                assert pending == set()
            assert set(records) == expected_tracked
            assert {
                saved for saved, path in paths.items() if path.is_file()
            } == expected_local
            assert service.live_steps == expected_remote
            assert_aliases(service, history, top_k, step)
