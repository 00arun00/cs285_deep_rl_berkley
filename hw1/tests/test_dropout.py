"""Dropout is reproducible, gated by training mode, and disabled by default."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hw1_imitation import evaluation, train
from hw1_imitation.data import Episode, EpisodesDataset
from hw1_imitation.model import BasePolicy, MSEPolicy, PolicyConfig, build_policy
from hw1_imitation.model import functional_dropout
from hw1_imitation.randomness import RandomStreamFactory, StreamId

from .test_randomness import TinyEnv


def make_policy(policy_type: str = "mse") -> BasePolicy:
    """Build a small deterministic-architecture policy without global RNG use."""
    policy = build_policy(
        PolicyConfig(
            policy_type=policy_type,  # type: ignore[arg-type]
            state_dim=3,
            action_dim=2,
            chunk_size=1,
            hidden_dims=(8, 8),
        ),
        cpu_generator=RandomStreamFactory(42).torch(StreamId.MODEL_INIT),
        target_device="cpu",
    )
    return policy


def make_batch(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(6, 3)).astype(np.float32)
    action = rng.normal(size=(6, 2)).astype(np.float32)
    return torch.from_numpy(state), torch.from_numpy(action)


class TestFunctionalDropout:
    def test_zero_probability_draws_nothing_and_matches_plain_path(self) -> None:
        """dropout_p = 0 preserves exact values and the caller's generator."""
        policy = make_policy()
        state, action_chunk = make_batch()
        generator = RandomStreamFactory(42).torch(StreamId.TRAIN_LOSS)
        before = generator.get_state().clone()

        with_dropout_zero = policy.compute_loss(
            state, action_chunk, generator=generator, dropout_p=0.0
        )
        without_flag = policy.compute_loss(state, action_chunk, generator=generator)
        assert torch.equal(with_dropout_zero, without_flag)
        expected = torch.nn.functional.mse_loss(
            policy.model(state), action_chunk.flatten(1)
        )
        assert torch.equal(with_dropout_zero, expected)
        assert torch.equal(generator.get_state(), before)

    def test_same_identity_replays_and_retained_generator_advances(self) -> None:
        """Equal identities replay the mask; a retained generator moves on."""
        policy = make_policy()
        state, action_chunk = make_batch()

        def loss(generator):
            return policy.compute_loss(
                state, action_chunk, generator=generator, dropout_p=0.5
            )

        first = RandomStreamFactory(42).torch(StreamId.TRAIN_LOSS, variation=3)
        replay = RandomStreamFactory(42).torch(StreamId.TRAIN_LOSS, variation=3)
        first_loss = loss(first)
        torch.testing.assert_close(first_loss, loss(replay), rtol=0, atol=0)
        assert not torch.equal(first_loss, loss(first))

    def test_mask_matches_manual_inverted_dropout(self) -> None:
        """functional_dropout reproduces inverted-dropout semantics."""
        x = torch.linspace(-1, 1, 12).reshape(3, 4)
        generator = RandomStreamFactory(9).torch(StreamId.TRAIN_LOSS)
        output = functional_dropout(x, dropout_p=0.5, generator=generator)
        kept = output != 0
        assert torch.equal(output[kept] / x[kept], torch.full_like(output[kept], 2.0))

    def test_invalid_probability_raises(self) -> None:
        x = torch.zeros(2, 2)
        generator = RandomStreamFactory(1).torch(StreamId.TRAIN_LOSS)
        for bad in (-0.1, 1.0):
            with pytest.raises(ValueError, match="dropout_p"):
                functional_dropout(x, dropout_p=bad, generator=generator)


class TestPolicyDropout:
    def test_mse_dropout_changes_loss_but_keeps_inference_deterministic(self) -> None:
        """Dropout applies inside compute_loss only; sampling stays unchanged."""
        policy = make_policy()
        assert isinstance(policy, MSEPolicy)
        state, action_chunk = make_batch()

        plain = policy.compute_loss(state, action_chunk, generator=torch.Generator())
        dropped = policy.compute_loss(
            state,
            action_chunk,
            generator=RandomStreamFactory(1).torch(StreamId.TRAIN_LOSS),
            dropout_p=0.5,
        )
        assert not torch.equal(plain, dropped)

        policy.train()
        training_actions = policy.sample_actions(state, generator=torch.Generator())
        policy.eval()
        eval_actions = policy.sample_actions(state, generator=torch.Generator())
        assert torch.equal(training_actions, eval_actions)

    def test_flow_dropout_requires_generator_and_replays(self) -> None:
        """Flow dropout threads the generator; missing generator raises."""
        policy = make_policy("flow")
        state, action_chunk = make_batch()

        with pytest.raises(ValueError, match="generator"):
            policy._forward(
                state,
                torch.zeros(6, 2),
                torch.zeros(6, 1),
                dropout_p=0.5,
            )

        def loss(generator):
            return policy.compute_loss(
                state, action_chunk, generator=generator, dropout_p=0.25
            )

        first = RandomStreamFactory(5).torch(StreamId.TRAIN_LOSS)
        replay = RandomStreamFactory(5).torch(StreamId.TRAIN_LOSS)
        torch.testing.assert_close(loss(first), loss(replay), rtol=0, atol=0)

    @pytest.mark.parametrize("policy_type", ["mse", "flow"])
    def test_eval_mode_disables_dropout_even_when_requested(self, policy_type):
        """Eval mode disables dropout without adding random draws to the loss."""
        policy = make_policy(policy_type)
        policy.eval()
        state, action_chunk = make_batch()
        plain_generator = RandomStreamFactory(6).torch(StreamId.TRAIN_LOSS)
        requested_generator = RandomStreamFactory(6).torch(StreamId.TRAIN_LOSS)
        global_before = torch.get_rng_state().clone()
        plain = policy.compute_loss(state, action_chunk, generator=plain_generator)
        requested = policy.compute_loss(
            state, action_chunk, generator=requested_generator, dropout_p=0.5
        )
        assert torch.equal(requested, plain)
        assert torch.equal(requested_generator.get_state(), plain_generator.get_state())
        assert torch.equal(torch.get_rng_state(), global_before)

    def test_train_mode_restores_dropout_application(self) -> None:
        """Returning to train() mode re-enables requested dropout."""
        policy = make_policy()
        state, action_chunk = make_batch()
        plain = policy.compute_loss(state, action_chunk, generator=torch.Generator())

        policy.eval()
        eval_loss = policy.compute_loss(
            state, action_chunk, generator=torch.Generator(), dropout_p=0.5
        )
        assert torch.equal(eval_loss, plain)

        policy.train()
        generator = RandomStreamFactory(8).torch(StreamId.TRAIN_LOSS)
        train_losses = [
            policy.compute_loss(
                state,
                action_chunk,
                generator=generator,
                dropout_p=0.5,
            )
            for _ in range(2)
        ]
        assert not torch.equal(train_losses[0], plain)
        assert not torch.equal(train_losses[0], train_losses[1])

    def test_dropout_gradients_flow_through_kept_units(self) -> None:
        """Backward propagation succeeds and updates every parameter tensor."""
        policy = make_policy()
        state, action_chunk = make_batch()
        loss = policy.compute_loss(
            state,
            action_chunk,
            generator=RandomStreamFactory(2).torch(StreamId.TRAIN_LOSS),
            dropout_p=0.3,
        )
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in policy.parameters()
        )


class TestConfigValidation:
    @pytest.mark.parametrize("value", [-0.1, 1.0, float("nan"), float("inf")])
    def test_invalid_dropout_values_raise(self, value):
        with pytest.raises(ValueError, match="dropout_p"):
            train.TrainConfig(dropout_p=value)


class TestCli:
    def test_flags_parse_in_both_syntaxes(self):
        assert train.parse_train_config(["--dropout-p", "0.2"]).dropout_p == 0.2
        assert train.parse_train_config(["--dropout_p=0.3"]).dropout_p == 0.3

    def test_defaults_keep_dropout_off(self):
        assert train.parse_train_config([]).dropout_p == 0.0


class TestEndToEnd:
    @pytest.mark.parametrize("policy_type", ["mse", "flow"])
    def test_dropout_training_is_reproducible_and_runs(self, tmp_path, policy_type):
        """Full driver runs with dropout and replays identically.

        Only external data/environment boundaries are patched; dropout masks
        and every other draw flow through the real streams.
        """
        rng = np.random.default_rng(11)
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

        def run(root: Path):
            losses: list[float] = []
            real_step = train.train_step

            def step(model, *args, **kwargs):
                loss = real_step(model, *args, **kwargs)
                losses.append(loss.item())
                return loss

            config = train.TrainConfig(
                policy_type=policy_type,
                hidden_dims=(4,),
                chunk_size=1,
                num_epochs=2,
                batch_size=4,
                validation_interval=100,
                eval_interval=100,
                eval_episodes=2,
                final_eval_episodes=2,
                flow_num_steps=2,
                num_video_episodes=0,
                save_checkpoints=False,
                log_wandb=False,
                log_csv=False,
                show_summary=False,
                dropout_p=0.3,
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
                train.run_training(config)
            assert len(losses) == 12
            assert all(np.isfinite(losses))
            return losses

        first = run(tmp_path / "first")
        second = run(tmp_path / "second")
        assert first == second
