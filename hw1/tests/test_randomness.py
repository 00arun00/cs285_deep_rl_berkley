"""Behavioral guarantees for random streams and their CPU consumers.

These tests check reproducibility and isolation, not statistical independence
or equivalence across devices and library versions.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from hw1_imitation import evaluation, train
from hw1_imitation.checkpoint import load_policy, save_policy
from hw1_imitation.data import Episode, EpisodesDataset, Normalizer
from hw1_imitation.model import BasePolicy, PolicyConfig, build_policy
from hw1_imitation.randomness import RandomStreamFactory, StreamId


def make_policy() -> BasePolicy:
    """Build a small stochastic policy without changing the default RNG."""
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.set_rng_state(
            RandomStreamFactory(42).torch(StreamId.MODEL_INIT).get_state()
        )
        return build_policy(
            PolicyConfig(
                policy_type="flow",
                state_dim=3,
                action_dim=2,
                chunk_size=1,
                hidden_dims=(4,),
            )
        )


def make_normalizer() -> Normalizer:
    return Normalizer(
        np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )


class TinyEnv(gym.Env):
    """Seeded observations and recorded actions, with no rendering or physics."""

    def __init__(self, first_episode_length: int = 2) -> None:
        self.action_space = gym.spaces.Box(-1e6, 1e6, shape=(2,), dtype=np.float32)
        self.first_episode_length = first_episode_length
        self.actions: list[list[np.ndarray]] = []
        self.initial_observations: list[np.ndarray] = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.np_random.normal(size=3).astype(np.float32)
        self.initial_observations.append(observation.copy())
        self.actions.append([])
        return observation, {}

    def step(self, action):
        self.actions[-1].append(action.copy())
        length = self.first_episode_length if len(self.actions) == 1 else 2
        done = len(self.actions[-1]) >= length
        observation = self.np_random.normal(size=3).astype(np.float32)
        return observation, 0.0, done, False, {}


class RandomStreamTests(unittest.TestCase):
    def test_existing_seed_identities_remain_stable(self) -> None:
        # These are compatibility fixtures, not recomputed expected values.
        # Changing an ID or encoding must not silently change existing runs.
        self.assertEqual(
            RandomStreamFactory(42).seed(StreamId.DATA_SPLIT),
            18311551174447640683,
        )
        self.assertEqual(
            RandomStreamFactory(2**80 + 42).seed(
                StreamId.EVAL_ENV, variation=3, index=7
            ),
            9736090538303049478,
        )

    def test_fresh_generators_replay_and_retained_generators_advance(self) -> None:
        streams = RandomStreamFactory(42)
        for backend in ("numpy", "torch"):
            with self.subTest(backend=backend):
                create = getattr(streams, backend)

                def draw(generator):
                    if backend == "numpy":
                        return generator.standard_normal(8)
                    return torch.randn(8, generator=generator).numpy()

                first = create(StreamId.TRAIN_LOSS, variation=3)
                replay = create(StreamId.TRAIN_LOSS, variation=3)
                first_draw = draw(first)
                np.testing.assert_array_equal(first_draw, draw(replay))
                second_draw = draw(first)
                self.assertFalse(np.array_equal(first_draw, second_draw))
                np.testing.assert_array_equal(second_draw, draw(replay))

    def test_episode_identities_survive_reordering_and_unrelated_draws(self) -> None:
        streams = RandomStreamFactory(42)

        def episode(index: int):
            return (
                streams.seed(StreamId.EVAL_ENV, variation=2, index=index),
                torch.randn(
                    8,
                    generator=streams.torch(
                        StreamId.EVAL_POLICY, variation=4, index=index
                    ),
                ),
            )

        expected = {index: episode(index) for index in (0, 1, 7)}
        streams.numpy(StreamId.DATA_SPLIT, variation=19).random(100)
        torch.randn(100, generator=streams.torch(StreamId.TRAIN_LOSS))
        for index in (7, 0, 1):
            seed, noise = episode(index)
            self.assertEqual(seed, expected[index][0])
            torch.testing.assert_close(noise, expected[index][1], rtol=0, atol=0)

        # Catch an accidentally ignored coordinate; this is not a proof of
        # collision freedom or statistical independence.
        seeds = [
            RandomStreamFactory(root).seed(stream, variation=variation, index=index)
            for root, stream, variation, index in (
                (42, StreamId.EVAL_ENV, 2, 7),
                (43, StreamId.EVAL_ENV, 2, 7),
                (42, StreamId.EVAL_POLICY, 2, 7),
                (42, StreamId.EVAL_ENV, 3, 7),
                (42, StreamId.EVAL_ENV, 2, 8),
            )
        ]
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_factory_consumers_leave_default_rngs_unchanged(self) -> None:
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        streams = RandomStreamFactory(42)
        streams.numpy(StreamId.DATA_SPLIT).random(32)
        torch.randn(32, generator=streams.torch(StreamId.TRAIN_LOSS))
        streams.seed(StreamId.EVAL_ENV, index=7)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_invalid_identities_fail_instead_of_being_coerced(self) -> None:
        for value, error in ((True, TypeError), (1.5, TypeError), (-1, ValueError)):
            with self.subTest(root=value), self.assertRaises(error):
                RandomStreamFactory(value)
        streams = RandomStreamFactory(42)
        for method in (streams.validate, streams.seed, streams.numpy, streams.torch):
            with self.subTest(method=method.__name__, stream="raw ID"):
                with self.assertRaises(TypeError):
                    method(int(StreamId.TRAIN_LOSS))
            for coordinate in ("variation", "index"):
                for value, error in (
                    (True, TypeError),
                    (1.5, TypeError),
                    (-1, ValueError),
                    (2**32, ValueError),
                ):
                    with (
                        self.subTest(
                            method=method.__name__, coordinate=coordinate, value=value
                        ),
                        self.assertRaises(error),
                    ):
                        method(StreamId.TRAIN_LOSS, **{coordinate: value})


class RandomnessIntegrationTests(unittest.TestCase):
    def test_loading_preserves_default_rng_even_when_weights_are_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            save_policy(path, make_policy(), make_normalizer())
            before = torch.get_rng_state().clone()
            load_policy(path)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

            payload = torch.load(path, weights_only=True)
            payload["model_state_dict"] = {}
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                load_policy(path)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_later_episode_actions_do_not_depend_on_earlier_episode_length(
        self,
    ) -> None:
        model = make_policy()

        def rollout(first_length: int) -> TinyEnv:
            env = TinyEnv(first_episode_length=first_length)
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
                )
            return env

        short, long = rollout(1), rollout(4)
        np.testing.assert_array_equal(
            short.initial_observations[1], long.initial_observations[1]
        )
        np.testing.assert_array_equal(short.actions[1], long.actions[1])

    def test_monitoring_frequency_does_not_change_flow_training(self) -> None:
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
        real_step = train.train_step

        def run(root: Path, validation_interval: int, eval_interval: int):
            losses: list[float] = []
            weights: dict[str, torch.Tensor] = {}

            def step(model, *args, **kwargs):
                loss = real_step(model, *args, **kwargs)
                losses.append(loss.item())
                weights.update(
                    {
                        name: value.detach().clone()
                        for name, value in model.state_dict().items()
                    }
                )
                return loss

            config = train.TrainConfig(
                policy_type="flow",
                hidden_dims=(4,),
                chunk_size=1,
                num_epochs=2,
                batch_size=4,
                validation_interval=validation_interval,
                eval_interval=eval_interval,
                eval_episodes=2,
                final_eval_episodes=2,
                flow_num_steps=2,
                num_video_episodes=0,
                save_checkpoints=False,
                log_wandb=False,
                log_csv=False,
                show_summary=False,
            )
            with (
                patch.object(train, "LOGDIR_PREFIX", str(root)),
                patch.object(train, "download_pusht", return_value=root),
                patch.object(train, "load_episodes_dataset", return_value=episodes),
                patch.object(
                    train.torch.accelerator, "current_accelerator", return_value=None
                ),
                patch.object(
                    evaluation.gym,
                    "make",
                    side_effect=lambda *args, **kwargs: TinyEnv(),
                ),
                patch.object(train, "train_step", side_effect=step),
            ):
                # Run real training, validation, and rollout evaluation. Only
                # external data/environment boundaries and observation are patched.
                train.run_training(config)
            return losses, weights

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_losses, expected_weights = run(root / "baseline", 100, 100)
            self.assertEqual(len(expected_losses), 12)
            for name, validation_interval, eval_interval in (
                ("more_validation", 1, 100),
                ("more_evaluation", 100, 1),
            ):
                with self.subTest(name=name):
                    losses, weights = run(
                        root / name, validation_interval, eval_interval
                    )
                    self.assertEqual(losses, expected_losses)
                    for key in expected_weights:
                        torch.testing.assert_close(
                            weights[key], expected_weights[key], rtol=0, atol=0
                        )
