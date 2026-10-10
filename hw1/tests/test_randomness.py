"""Behavioral guarantees for random streams and their CPU consumers.

These tests check reproducibility and isolation, not statistical independence
or equivalence across devices and library versions.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch
from hw1_imitation import evaluation, train
from hw1_imitation.checkpoint import load_policy, save_policy
from hw1_imitation.data import Episode, EpisodesDataset, Normalizer
from hw1_imitation.model import BasePolicy, PolicyConfig, SimpleMLP, build_policy
from hw1_imitation.randomness import RandomStreamFactory, StreamId
from hypothesis import example, given
from hypothesis import strategies as st


def make_policy() -> BasePolicy:
    """Build a small stochastic policy without changing the default RNG."""
    return build_policy(
        PolicyConfig(
            policy_type="flow",
            state_dim=3,
            action_dim=2,
            chunk_size=1,
            hidden_dims=(4,),
        ),
        cpu_generator=RandomStreamFactory(42).torch(StreamId.MODEL_INIT),
        target_device="cpu",
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


class TestRandomStream:
    def test_existing_seed_identities_remain_stable(self) -> None:
        # These are compatibility fixtures, not recomputed expected values.
        # Changing an ID or encoding must not silently change existing runs.
        assert RandomStreamFactory(42).seed(StreamId.DATA_SPLIT) == 18311551174447640683
        assert (
            RandomStreamFactory(2**80 + 42).seed(
                StreamId.EVAL_ENV, variation=3, index=7
            )
            == 9736090538303049478
        )

    @pytest.mark.parametrize("backend", ("numpy", "torch"))
    def test_fresh_generators_replay_and_retained_generators_advance(
        self, backend
    ) -> None:
        streams = RandomStreamFactory(42)
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
        assert not np.array_equal(first_draw, second_draw)
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
            assert seed == expected[index][0]
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
        assert len(set(seeds)) == len(seeds)

    def test_factory_consumers_leave_default_rngs_unchanged(self) -> None:
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        streams = RandomStreamFactory(42)
        streams.numpy(StreamId.DATA_SPLIT).random(32)
        torch.randn(32, generator=streams.torch(StreamId.TRAIN_LOSS))
        streams.seed(StreamId.EVAL_ENV, index=7)
        numpy_after = np.random.get_state()
        assert numpy_before[0] == numpy_after[0]
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        assert numpy_before[2:] == numpy_after[2:]
        assert torch.equal(torch_before, torch.get_rng_state())

    @pytest.mark.parametrize(
        "value,error",
        [
            (True, TypeError),
            (1.5, TypeError),
            (None, TypeError),
            ("42", TypeError),
            (np.int64(42), TypeError),
            (-1, ValueError),
        ],
    )
    def test_invalid_root_seeds_fail_instead_of_being_coerced(self, value, error):
        with pytest.raises(error):
            RandomStreamFactory(value)

    @given(root=st.integers(min_value=0, max_value=2**256))
    @example(root=0)
    @example(root=2**80 + 42)
    def test_valid_root_seeds_are_preserved(self, root):
        factory = RandomStreamFactory(root)
        assert factory.root_seed == root
        seed = factory.seed(StreamId.TRAIN_LOSS)
        assert 0 <= seed < 2**64
        assert seed == RandomStreamFactory(root).seed(StreamId.TRAIN_LOSS)

    @given(root=st.integers(max_value=-1))
    def test_negative_root_seeds_are_rejected(self, root):
        with pytest.raises(ValueError):
            RandomStreamFactory(root)

    @pytest.mark.parametrize("method_name", ["validate", "seed", "numpy", "torch"])
    def test_raw_stream_ids_are_rejected(self, method_name):
        method = getattr(RandomStreamFactory(42), method_name)
        with pytest.raises(TypeError):
            method(int(StreamId.TRAIN_LOSS))

    @pytest.mark.parametrize("method_name", ["validate", "seed", "numpy", "torch"])
    @pytest.mark.parametrize("coordinate", ["variation", "index"])
    @pytest.mark.parametrize(
        "value,error",
        [
            (True, TypeError),
            (1.5, TypeError),
            (None, TypeError),
            ("0", TypeError),
            (np.int64(0), TypeError),
            (-1, ValueError),
            (2**32, ValueError),
        ],
    )
    def test_invalid_coordinates_fail_instead_of_being_coerced(
        self, method_name, coordinate, value, error
    ):
        method = getattr(RandomStreamFactory(42), method_name)
        with pytest.raises(error):
            method(StreamId.TRAIN_LOSS, **{coordinate: value})

    @pytest.mark.parametrize("method_name", ["validate", "seed", "numpy", "torch"])
    @given(variation=st.integers(0, 2**32 - 1), index=st.integers(0, 2**32 - 1))
    @example(variation=0, index=0)
    @example(variation=2**32 - 1, index=2**32 - 1)
    def test_valid_coordinate_boundaries_are_accepted(
        self, method_name, variation, index
    ):
        method = getattr(RandomStreamFactory(42), method_name)
        method(StreamId.TRAIN_LOSS, variation=variation, index=index)

    @pytest.mark.parametrize("method_name", ["validate", "seed", "numpy", "torch"])
    @pytest.mark.parametrize("coordinate", ["variation", "index"])
    @given(value=st.one_of(st.integers(max_value=-1), st.integers(min_value=2**32)))
    def test_out_of_range_coordinates_are_rejected(
        self, method_name, coordinate, value
    ):
        method = getattr(RandomStreamFactory(42), method_name)
        with pytest.raises(ValueError):
            method(StreamId.TRAIN_LOSS, **{coordinate: value})


class TestPolicyInitialization:
    @pytest.mark.parametrize("policy_type", ("mse", "flow"))
    def test_concurrent_initialization_matches_isolated_runs(self, policy_type) -> None:
        config = PolicyConfig(
            policy_type=policy_type,
            state_dim=3,
            action_dim=2,
            chunk_size=1,
            hidden_dims=(4,),
        )

        def build(variation: int) -> BasePolicy:
            return build_policy(
                config,
                cpu_generator=RandomStreamFactory(42).torch(
                    StreamId.MODEL_INIT,
                    variation=variation,
                ),
                target_device="cpu",
            )

        before = torch.get_rng_state().clone()
        expected = [build(variation) for variation in (0, 1)]
        assert torch.equal(before, torch.get_rng_state())

        barrier = Barrier(2)

        def concurrent_build(variation: int) -> BasePolicy:
            barrier.wait(timeout=10)
            return build(variation)

        with ThreadPoolExecutor(max_workers=2) as executor:
            actual = list(executor.map(concurrent_build, (0, 1)))

        assert torch.equal(before, torch.get_rng_state())

        for reference, result in zip(expected, actual):
            assert result.training
            for name, value in result.state_dict().items():
                assert value.device == torch.device("cpu")
                assert torch.isfinite(value).all().item()
                torch.testing.assert_close(
                    value,
                    reference.state_dict()[name],
                    rtol=0,
                    atol=0,
                )

        assert any(
            (
                not torch.equal(value, expected[1].state_dict()[name])
                for name, value in expected[0].state_dict().items()
            )
        )

    @pytest.mark.parametrize("policy_type", ["mse", "flow"])
    def test_initialization_advances_only_the_supplied_generator(
        self, policy_type
    ) -> None:
        generator = RandomStreamFactory(42).torch(StreamId.MODEL_INIT)
        generator_before = generator.get_state().clone()
        default_before = torch.get_rng_state().clone()

        build_policy(
            PolicyConfig(
                policy_type=policy_type,
                state_dim=3,
                action_dim=2,
                chunk_size=1,
                hidden_dims=(4,),
            ),
            cpu_generator=generator,
        )

        assert not torch.equal(generator_before, generator.get_state())
        assert torch.equal(default_before, torch.get_rng_state())

    def test_failed_construction_preserves_default_rng(self) -> None:
        before = torch.get_rng_state().clone()

        with pytest.raises(ValueError):
            build_policy(
                PolicyConfig(
                    policy_type="mse",
                    state_dim=0,
                    action_dim=2,
                    chunk_size=1,
                ),
                cpu_generator=RandomStreamFactory(42).torch(
                    StreamId.MODEL_INIT,
                ),
            )

        assert torch.equal(before, torch.get_rng_state())

    @pytest.mark.parametrize("policy_type", ["mse", "flow"])
    def test_failed_initialization_preserves_default_rng(self, policy_type):
        generator = RandomStreamFactory(42).torch(StreamId.MODEL_INIT)
        generator_before = generator.get_state().clone()
        global_before = torch.get_rng_state().clone()
        real_reset = SimpleMLP.reset_parameters

        def fail_after_initialization(model, *, generator):
            real_reset(model, generator=generator)
            raise RuntimeError("injected initialization failure")

        with patch.object(SimpleMLP, "reset_parameters", fail_after_initialization):
            with pytest.raises(RuntimeError, match="injected initialization failure"):
                build_policy(
                    PolicyConfig(policy_type, 3, 2, 1, (4,)), cpu_generator=generator
                )
        assert not torch.equal(generator_before, generator.get_state())
        assert torch.equal(global_before, torch.get_rng_state())

    def test_target_device_transfer_preserves_initialized_values(self) -> None:
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is None or not torch.accelerator.is_available():
            pytest.skip("No accelerator available")

        config = PolicyConfig(
            policy_type="flow",
            state_dim=3,
            action_dim=2,
            chunk_size=1,
            hidden_dims=(4,),
        )

        def build(device: torch.device | str) -> BasePolicy:
            return build_policy(
                config,
                cpu_generator=RandomStreamFactory(42).torch(
                    StreamId.MODEL_INIT,
                ),
                target_device=device,
            )

        reference = build("cpu")
        actual = build(accelerator)

        for name, value in actual.state_dict().items():
            assert value.device.type == accelerator.type
            torch.testing.assert_close(
                value.cpu(),
                reference.state_dict()[name],
                rtol=0,
                atol=0,
            )


class TestRandomnessIntegration:
    def test_loading_and_initialization_can_run_concurrently(self, tmp_path) -> None:
        path = tmp_path / "policy.pt"
        reference = make_policy()
        initialized_reference = make_policy()
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.add_(1.0)
        save_policy(path, reference, make_normalizer())

        before = torch.get_rng_state().clone()
        barrier = Barrier(2)

        def initialize() -> BasePolicy:
            barrier.wait(timeout=10)
            return make_policy()

        def load() -> BasePolicy:
            barrier.wait(timeout=10)
            model, _, _ = load_policy(path)
            return model

        with ThreadPoolExecutor(max_workers=2) as executor:
            initialized_future = executor.submit(initialize)
            loaded_future = executor.submit(load)
            initialized = initialized_future.result()
            loaded = loaded_future.result()

        assert torch.equal(before, torch.get_rng_state())
        assert initialized.training
        assert not loaded.training

        for result, expected in (
            (initialized, initialized_reference),
            (loaded, reference),
        ):
            for name, value in result.state_dict().items():
                torch.testing.assert_close(
                    value,
                    expected.state_dict()[name],
                    rtol=0,
                    atol=0,
                )

    def test_loading_preserves_default_rng_even_when_weights_are_invalid(
        self, tmp_path
    ) -> None:
        path = tmp_path / "policy.pt"
        save_policy(path, make_policy(), make_normalizer())
        before = torch.get_rng_state().clone()
        load_policy(path)
        assert torch.equal(before, torch.get_rng_state())

        payload = torch.load(path, weights_only=True)
        payload["model_state_dict"] = {}
        torch.save(payload, path)
        with pytest.raises(RuntimeError):
            load_policy(path)
        assert torch.equal(before, torch.get_rng_state())

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

    @pytest.mark.parametrize(
        "name,validation_interval,eval_interval",
        (("more_validation", 1, 100), ("more_evaluation", 100, 1)),
    )
    def test_monitoring_frequency_does_not_change_flow_training(
        self, name, validation_interval, eval_interval, tmp_path
    ) -> None:
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
                patch.object(
                    train,
                    "compute_validation_loss",
                    wraps=train.compute_validation_loss,
                ) as validate,
                patch.object(
                    train, "evaluate_policy", wraps=train.evaluate_policy
                ) as evaluate,
            ):
                # Run real training, validation, and rollout evaluation. Only
                # external data/environment boundaries and observation are patched.
                train.run_training(config)
            expected_validation_calls = sum(
                step % validation_interval == 0 or step == 12 for step in range(1, 13)
            )
            expected_evaluation_calls = sum(
                step % eval_interval == 0 or step == 12 for step in range(1, 13)
            )
            assert validate.call_count == expected_validation_calls
            assert evaluate.call_count == expected_evaluation_calls
            return losses, weights

        root = tmp_path
        expected_losses, expected_weights = run(root / "baseline", 100, 100)
        assert len(expected_losses) == 12
        losses, weights = run(root / name, validation_interval, eval_interval)
        assert losses == expected_losses
        for key in expected_weights:
            torch.testing.assert_close(
                weights[key], expected_weights[key], rtol=0, atol=0
            )
