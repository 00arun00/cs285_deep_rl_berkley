"""Retention behavior with local files and a simulated W&B artifact service."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from hw1_imitation import evaluation


class CheckpointRetentionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
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

            def wait():
                # No existing retained file may disappear before upload succeeds.
                for record in self.checkpoints.values():
                    if record.path.exists():
                        self.assertEqual(record.path.read_bytes(), b"checkpoint")
                if self.upload_error:
                    raise self.upload_error
                for alias in aliases:
                    self.aliases[alias] = step

            def delete():
                self.assertNotIn(step, self.aliases.values())

            artifact.wait.side_effect = wait
            artifact.delete.side_effect = delete
            return artifact

        self.run.log_artifact.side_effect = log_artifact
        for patcher in (
            patch.object(evaluation.wandb, "run", self.run),
            patch.object(
                evaluation.wandb, "Artifact", side_effect=lambda **kw: Mock(**kw)
            ),
            patch.object(evaluation, "save_policy", side_effect=save_policy),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def save(self, step, score, top_k=3):
        return evaluation.log_checkpoint_artifact(
            Mock(),
            step,
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
        for step, score in enumerate([0.9, 0.8, 0.7, 0.1, 0.7], start=1):
            self.save(step, score)
        self.assertEqual(set(self.checkpoints), {1, 2, 3, 5})
        self.assertEqual(self.local_steps(), {1, 2, 3, 5})
        self.artifacts[4].delete.assert_called_once_with()
        self.assertEqual(self.aliases, {"best": 1, "latest": 5})

        self.save(6, 1.0)
        self.assertEqual(set(self.checkpoints), {1, 2, 6})
        self.assertEqual(self.local_steps(), {1, 2, 6})
        self.assertEqual(self.aliases, {"best": 6, "latest": 6})

    def test_zero_keeps_only_latest(self):
        self.save(1, 1.0, top_k=0)
        self.save(2, 0.1, top_k=0)
        self.assertEqual(set(self.checkpoints), {2})
        self.assertEqual(self.local_steps(), {2})
        self.assertEqual(self.aliases, {"latest": 2})

    def test_failed_upload_does_not_prune(self):
        self.save(1, 1.0, top_k=0)
        self.upload_error = RuntimeError("upload failed")
        with self.assertRaisesRegex(RuntimeError, "upload failed"):
            self.save(2, 0.1, top_k=0)
        self.assertEqual(set(self.checkpoints), {1})
        self.assertEqual(self.local_steps(), {1, 2})
        self.artifacts[1].delete.assert_not_called()

    def test_failed_remote_cleanup_retries_without_retaining_local_file(self):
        self.save(1, 1.0, top_k=0)
        self.artifacts[1].delete.side_effect = RuntimeError("network unavailable")
        with self.assertLogs(level="ERROR"):
            self.assertEqual(self.save(2, 0.1, top_k=0), {1})
        self.assertEqual(set(self.checkpoints), {1, 2})
        self.assertEqual(self.local_steps(), {2})
        self.artifacts[1].delete.side_effect = None
        self.assertEqual(self.save(3, 0.2, top_k=0), set())
        self.assertEqual(set(self.checkpoints), {3})
        self.assertEqual(self.artifacts[1].delete.call_count, 2)

    def test_failed_local_cleanup_retries_before_remote_deletion(self):
        self.save(1, 1.0, top_k=0)
        with (
            patch.object(Path, "unlink", side_effect=OSError("permission denied")),
            self.assertLogs(level="ERROR"),
        ):
            self.assertEqual(self.save(2, 0.1, top_k=0), {1})
        self.assertEqual(self.local_steps(), {1, 2})
        self.artifacts[1].delete.assert_not_called()

        self.assertEqual(self.save(3, 0.2, top_k=0), set())
        self.assertEqual(self.local_steps(), {3})
        self.assertEqual(set(self.checkpoints), {3})
        self.artifacts[1].delete.assert_called_once_with()

    def test_invalid_inputs_do_not_save(self):
        for score in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                self.save(1, score)
        self.run.offline = True
        with self.assertRaisesRegex(ValueError, "online"):
            self.save(1, 0.5)
        self.assertFalse((self.root / "checkpoints").exists())
