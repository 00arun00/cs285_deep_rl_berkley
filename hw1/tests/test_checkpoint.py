"""CPU tests for reusable policy checkpoint components."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hw1_imitation.checkpoint import load_policy, save_policy
from hw1_imitation.data import Normalizer
from hw1_imitation.model import PolicyConfig, build_policy


class CheckpointTests(unittest.TestCase):
    def test_normalizer_state_does_not_share_storage(self):
        normalizer = Normalizer(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([5.0], dtype=np.float32),
            np.array([6.0], dtype=np.float32),
        )
        state = normalizer.state_dict()
        restored = Normalizer.from_state_dict(state)
        for name, tensor in state.items():
            self.assertEqual(tensor.device.type, "cpu")
            np.testing.assert_array_equal(
                getattr(restored, name), getattr(normalizer, name)
            )
            tensor.fill_(99)
            np.testing.assert_array_equal(
                getattr(restored, name), getattr(normalizer, name)
            )
            self.assertFalse(
                np.shares_memory(getattr(restored, name), getattr(normalizer, name))
            )


    def test_normalizer_rejects_invalid_statistics(self):
        for name, value in (
            ("state_mean", torch.tensor([float("nan"), 0.0])),
            ("action_mean", torch.tensor([float("inf")])),
            ("state_std", torch.tensor([0.0, 1.0])),
            ("action_std", torch.tensor([-1.0])),
            ("action_std", torch.tensor([float("nan")])),
            ("state_std", torch.ones(3)),
            ("state_mean", torch.ones(1, 2)),
        ):
            with self.subTest(name=name, value=value):
                state = {
                    "state_mean": torch.zeros(2),
                    "state_std": torch.ones(2),
                    "action_mean": torch.zeros(1),
                    "action_std": torch.ones(1),
                }
                state[name] = value
                with self.assertRaises(ValueError):
                    Normalizer.from_state_dict(state)
                with self.assertRaises(ValueError):
                    Normalizer(**{key: tensor.numpy() for key, tensor in state.items()})


    def test_policy_round_trip_preserves_raw_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            for policy_type in ("mse", "flow"):
                with self.subTest(policy_type=policy_type):
                    model = build_policy(
                        PolicyConfig(
                            policy_type=policy_type,
                            state_dim=3,
                            action_dim=2,
                            chunk_size=2,
                            hidden_dims=(8,),
                        )
                    )
                    normalizer = Normalizer(
                        np.array([1.0, 2.0, 3.0], dtype=np.float32),
                        np.array([2.0, 3.0, 4.0], dtype=np.float32),
                        np.array([5.0, 6.0], dtype=np.float32),
                        np.array([7.0, 8.0], dtype=np.float32),
                    )
                    path = Path(directory) / policy_type / "policy.pt"
                    save_policy(path, model, normalizer, flow_num_steps=3)
                    restored, stats, inference = load_policy(path, device="cpu")

                    self.assertEqual(restored.config, model.config)
                    self.assertFalse(restored.training)
                    self.assertTrue(model.training)
                    self.assertEqual(inference, {"flow_num_steps": 3})
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    self.assertNotIn("resume", payload)
                    self.assertNotIn("optimizer_state_dict", payload)
                    self.assertNotIn("random_state", payload)
                    for key, value in model.state_dict().items():
                        self.assertEqual(
                            payload["model_state_dict"][key].device.type, "cpu"
                        )
                        torch.testing.assert_close(
                            restored.state_dict()[key],
                            value,
                            rtol=0,
                            atol=0,
                        )

                    raw = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)

                    def predict(policy, norm):
                        # Flow sampling is stochastic; compare with identical
                        # inference noise, not restored training RNG state.
                        torch.manual_seed(123)
                        with torch.no_grad():
                            actions = policy.sample_actions(
                                torch.from_numpy(norm.normalize_state(raw)),
                                num_steps=inference["flow_num_steps"],
                            )
                        return norm.denormalize_action(actions.numpy())

                    np.testing.assert_array_equal(
                        predict(model, normalizer),
                        predict(restored, stats),
                    )


    def test_reject_unknown_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.pt"
            torch.save({"format_version": 999}, path)
            with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
                load_policy(path)



if __name__ == "__main__":
    unittest.main()
