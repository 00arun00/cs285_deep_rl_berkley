"""Retention behavior with local files and a simulated W&B artifact service."""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from hw1_imitation import evaluation


class TestCheckpointRetention:
    @pytest.fixture(autouse=True)
    def retention_service(self, tmp_path, monkeypatch):
        self.root = tmp_path
        self.run = Mock(dir=str(self.root), id="test", offline=False)
        self.checkpoints = {}
        self.artifacts = {}
        self.aliases = {}
        self.upload_error = None

        def save_policy(*, path, **kwargs):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")

        def log_artifact(artifact, *, aliases):
            step = artifact.metadata["step"]
            self.artifacts[step] = artifact

            existing_paths = [
                record.path
                for record in self.checkpoints.values()
                if record.path.exists()
            ]

            def wait():
                # No existing retained file may disappear before upload succeeds.
                for path in existing_paths:
                    assert path.read_bytes() == b"checkpoint"
                if self.upload_error:
                    raise self.upload_error
                for alias in aliases:
                    self.aliases[alias] = step

            def delete():
                assert step not in self.aliases.values()

            artifact.wait.side_effect = wait
            artifact.delete.side_effect = delete
            return artifact

        self.run.log_artifact.side_effect = log_artifact
        monkeypatch.setattr(
            evaluation.wandb, "Artifact", Mock(side_effect=lambda **kw: Mock(**kw))
        )
        monkeypatch.setattr(evaluation, "save_policy", save_policy)

    def save(self, step, score, top_k=3):
        return evaluation.save_checkpoint_and_retain(
            Mock(),
            step,
            checkpoint_dir=self.root / "checkpoints",
            run=self.run,
            normalizer=Mock(),
            flow_num_steps=10,
            mean_reward=score,
            top_k=top_k,
            checkpoints=self.checkpoints,
        )

    def local_steps(self):
        return {
            int(path.stem.removeprefix("policy_step_"))
            for path in (self.root / "checkpoints").glob("*.pt")
        }

    def test_best_three_plus_latest_and_ties(self):
        """Retention preserves best policies and the latest recovery point.

        Protects:
            Earlier ties win and remote aliases follow the retained best and latest
            checkpoints.
        Value:
            Catches premature deletion of useful weights as scores improve or
            decline.
        Approach:
            Use a readable score history with real files and a simulated external
            artifact service.
        """
        for step, score in enumerate([0.9, 0.8, 0.7, 0.1, 0.7], start=1):
            self.save(step, score)
        assert set(self.checkpoints) == {1, 2, 3, 5}
        assert self.local_steps() == {1, 2, 3, 5}
        self.artifacts[4].delete.assert_called_once_with()
        assert self.aliases == {"best": 1, "latest": 5}

        self.save(6, 1.0)
        assert set(self.checkpoints) == {1, 2, 6}
        assert self.local_steps() == {1, 2, 6}
        assert self.aliases == {"best": 6, "latest": 6}

    def test_zero_keeps_only_latest(self):
        """Keeps the zero-retention option usable without accumulating files or a best
        alias."""
        self.save(1, 1.0, top_k=0)
        self.save(2, 0.1, top_k=0)
        assert set(self.checkpoints) == {2}
        assert self.local_steps() == {2}
        assert self.aliases == {"latest": 2}

    def test_failed_upload_does_not_prune(self):
        """Upload failure preserves the previously committed retention state.

        Protects:
            Existing files, aliases, and records survive until replacement upload
            completes.
        Value:
            Prevents losing the last remote recovery point during a failed upload.
        Approach:
            Fail the artifact completion boundary and observe local files and
            service calls.
        """
        self.save(1, 1.0, top_k=0)
        before = dict(self.checkpoints)
        aliases_before = dict(self.aliases)
        original_upload = self.run.log_artifact.side_effect

        def upload(artifact, *, aliases):
            assert self.checkpoints == before
            assert self.aliases == aliases_before
            assert before[1].path.read_bytes() == b"checkpoint"
            return original_upload(artifact, aliases=aliases)

        self.run.log_artifact.side_effect = upload
        self.upload_error = RuntimeError("upload failed")
        with pytest.raises(RuntimeError, match="upload failed"):
            self.save(2, 0.1, top_k=0)
        assert set(self.checkpoints) == {1}
        assert self.local_steps() == {1, 2}
        assert self.checkpoints == before
        assert self.aliases == aliases_before
        self.artifacts[1].delete.assert_not_called()

    def test_failed_remote_cleanup_retries_without_retaining_local_file(self, caplog):
        """Remote cleanup failure remains retryable after local deletion.

        Protects:
            Failed remote deletion stays tracked and is retried on the next save.
        Value:
            Prevents remote storage leaks from being forgotten after the local copy
            is removed.
        Approach:
            Toggle deletion failure in the service fake and inspect retry state
            through recovery.
        """
        self.save(1, 1.0, top_k=0)
        self.artifacts[1].delete.side_effect = RuntimeError("network unavailable")
        caplog.clear()
        with caplog.at_level("ERROR"):
            assert self.save(2, 0.1, top_k=0) == {1}
        assert any(record.levelno >= logging.ERROR for record in caplog.records)
        assert set(self.checkpoints) == {1, 2}
        assert self.local_steps() == {2}
        self.artifacts[1].delete.side_effect = None
        assert self.save(3, 0.2, top_k=0) == set()
        assert set(self.checkpoints) == {3}
        assert self.artifacts[1].delete.call_count == 2

    def test_failed_local_cleanup_retries_before_remote_deletion(self, caplog):
        """Local cleanup failures retain enough state for a later retry.

        Protects:
            Remote deletion waits for successful local cleanup and errors are
            reported.
        Value:
            Prevents cleanup failures from silently abandoning files or losing retry
            records.
        Approach:
            Inject a filesystem deletion error, restore the operation, and save
            again.
        """
        self.save(1, 1.0, top_k=0)
        caplog.clear()
        with (
            patch.object(Path, "unlink", side_effect=OSError("permission denied")),
            caplog.at_level("ERROR"),
        ):
            assert self.save(2, 0.1, top_k=0) == {1}
        assert any(record.levelno >= logging.ERROR for record in caplog.records)
        assert self.local_steps() == {1, 2}
        self.artifacts[1].delete.assert_not_called()

        assert self.save(3, 0.2, top_k=0) == set()
        assert self.local_steps() == {3}
        assert set(self.checkpoints) == {3}
        self.artifacts[1].delete.assert_called_once_with()

    @pytest.mark.parametrize("top_k,expected", ((0, {5}), (3, {1, 2, 3, 5})))
    def test_local_only_retention_does_not_use_global_wandb_run(self, top_k, expected):
        """Local retention remains independent of ambient W&B state.

        Protects:
            An unrelated global run cannot trigger uploads or alter the local
            retention budget.
        Value:
            Catches accidental network activity and artifacts being attached to
            another experiment.
        Approach:
            Install an ambient run sentinel while calling the public local-only save
            path.
        """
        self.checkpoints = {}
        checkpoint_dir = self.root / f"local-{top_k}"
        with patch.object(evaluation.wandb, "run", self.run):
            for step, score in enumerate([0.9, 0.8, 0.7, 0.1, 0.7], 1):
                pending = evaluation.save_checkpoint_and_retain(
                    Mock(),
                    step,
                    checkpoint_dir=checkpoint_dir,
                    normalizer=Mock(),
                    flow_num_steps=10,
                    mean_reward=score,
                    top_k=top_k,
                    checkpoints=self.checkpoints,
                )
                assert pending == set()
        assert set(self.checkpoints) == expected
        assert {
            int(p.stem.removeprefix("policy_step_"))
            for p in checkpoint_dir.glob("*.pt")
        } == expected
        assert all((r.artifact is None for r in self.checkpoints.values()))
        self.run.log_artifact.assert_not_called()
        evaluation.wandb.Artifact.assert_not_called()

    @pytest.mark.parametrize(
        "score,top_k,offline,message",
        [
            (float("nan"), 3, False, "finite"),
            (float("inf"), 3, False, "finite"),
            (-float("inf"), 3, False, "finite"),
            (0.5, -1, False, "top_k"),
            (0.5, 3, True, "online"),
        ],
    )
    def test_invalid_inputs_do_not_save(self, score, top_k, offline, message):
        """Ensures invalid retention requests fail before creating files or remote
        artifacts."""
        self.run.offline = offline
        with patch.object(evaluation, "save_policy") as save:
            with pytest.raises(ValueError, match=message):
                self.save(1, score, top_k=top_k)
            save.assert_not_called()
        self.run.log_artifact.assert_not_called()
        evaluation.wandb.Artifact.assert_not_called()
        assert self.checkpoints == {}
        assert self.aliases == {}
        assert not (self.root / "checkpoints").exists()
