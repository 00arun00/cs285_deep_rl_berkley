"""CPU tests for reusable policy checkpoint components."""

import unittest

import numpy as np
import torch

from hw1_imitation.data import Normalizer


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



if __name__ == "__main__":
    unittest.main()
