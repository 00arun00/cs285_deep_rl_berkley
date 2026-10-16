"""Explicit dropout placement, policy mode gating, and checkpoint configuration."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from hw1_imitation import evaluation, train
from hw1_imitation.checkpoint import load_policy, save_policy
from hw1_imitation.data import Episode, EpisodesDataset
from hw1_imitation.model import PolicyConfig, MLP, build_policy
from hw1_imitation.randomness import RandomStreamFactory, StreamId
from hw1_imitation.rng_modules import (
    DropoutWithExplicitRNG,
    get_dropout_config,
    override_dropout_p,
)
from tests.test_randomness import TinyEnv, make_normalizer


def make_policy(policy_type):
    return build_policy(
        PolicyConfig(
            policy_type, state_dim=3, action_dim=2, chunk_size=1, hidden_dims=(8, 8)
        ),
        cpu_generator=RandomStreamFactory(42).torch(StreamId.MODEL_INIT),
    )


def test_explicit_placement_preserves_legacy_parameter_names_and_forward():
    mlp = MLP(3, (8, 8), 2)
    assert [type(layer) for layer in mlp] == [
        nn.Linear,
        nn.ReLU,
        DropoutWithExplicitRNG,
        nn.Linear,
        nn.ReLU,
        DropoutWithExplicitRNG,
        nn.Linear,
    ]
    assert set(mlp.state_dict()) == {
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
        "4.weight",
        "4.bias",
    }
    x = torch.ones(2, 3)
    legacy = nn.Sequential(
        *(m for m in mlp if not isinstance(m, DropoutWithExplicitRNG))
    )
    generator = torch.Generator().manual_seed(5)
    rng_state_before = generator.get_state().clone()
    assert torch.equal(mlp(x, generator=generator), legacy(x))
    assert torch.equal(rng_state_before, generator.get_state())
    assert [type(m) for m in MLP(3, (), 2)] == [nn.Linear]


@pytest.mark.parametrize("policy_type", ["mse", "flow"])
def test_policy_dropout_replays_and_eval_removes_only_dropout_draws(policy_type):
    policy = make_policy(policy_type)
    override_dropout_p(policy, 0.5)
    state = torch.ones(16, 3)
    actions = torch.ones(16, 2)
    global_rng_state_before = torch.get_rng_state().clone()

    def loss():
        generator = torch.Generator().manual_seed(11)
        result = policy.compute_loss(state, actions, generator=generator)
        return result, generator.get_state()

    train_loss, train_rng_state = loss()
    replayed_loss, replayed_rng_state = loss()
    assert torch.equal(train_loss, replayed_loss)
    assert torch.equal(train_rng_state, replayed_rng_state)
    train_loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters()
    )

    policy.eval()
    eval_loss, eval_rng_state = loss()
    eval_actions = policy.sample_actions(
        state, generator=torch.Generator().manual_seed(3)
    )
    override_dropout_p(policy, 0)
    no_dropout_loss, no_dropout_rng_state = loss()
    no_dropout_actions = policy.sample_actions(
        state, generator=torch.Generator().manual_seed(3)
    )
    assert torch.equal(eval_loss, no_dropout_loss)
    assert torch.equal(eval_rng_state, no_dropout_rng_state)
    assert torch.equal(eval_actions, no_dropout_actions)
    assert not torch.equal(train_rng_state, no_dropout_rng_state)
    assert torch.equal(global_rng_state_before, torch.get_rng_state())
    override_dropout_p(policy, 0.5)
    policy.train()
    restored_train_loss, restored_rng_state = loss()
    assert torch.equal(train_loss, restored_train_loss)
    assert torch.equal(train_rng_state, restored_rng_state)


@pytest.mark.parametrize("policy_type", ["mse", "flow"])
def test_sampling_dropout_replays_and_follows_training_mode(policy_type):
    policy = make_policy(policy_type)
    override_dropout_p(policy, 0.5)
    states = torch.ones(16, policy.state_dim)
    global_rng_state_before = torch.get_rng_state().clone()

    def sample(steps):
        generator = torch.Generator().manual_seed(19)
        actions = policy.sample_actions(states, generator=generator, num_steps=steps)
        assert torch.isfinite(actions).all()
        return actions, generator.get_state()

    train_actions, train_rng_state = sample(2)
    replayed_actions, replayed_rng_state = sample(2)
    assert torch.equal(train_actions, replayed_actions)
    assert torch.equal(train_rng_state, replayed_rng_state)
    if policy_type == "flow":
        # Extra integration steps consume dropout randomness only in train mode.
        assert not torch.equal(train_rng_state, sample(5)[1])

    policy.eval()
    eval_actions, eval_rng_state = sample(2)
    assert not torch.equal(train_actions, eval_actions)
    assert not torch.equal(train_rng_state, eval_rng_state)
    if policy_type == "flow":
        assert torch.equal(eval_rng_state, sample(5)[1])
    override_dropout_p(policy, 0)
    no_dropout_actions, no_dropout_rng_state = sample(2)
    assert torch.equal(eval_actions, no_dropout_actions)
    assert torch.equal(eval_rng_state, no_dropout_rng_state)
    assert torch.equal(global_rng_state_before, torch.get_rng_state())


@pytest.mark.parametrize("value", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_dropout_rejected_before_training(value):
    with pytest.raises(ValueError, match="dropout_p"):
        train.TrainConfig(dropout_p=value)


def test_dropout_cli_default_and_overrides():
    assert train.parse_train_config([]).dropout_p is None
    assert train.parse_train_config(["--dropout-p", "0"]).dropout_p == 0
    assert train.parse_train_config(["--dropout-p", "0.2"]).dropout_p == 0.2
    assert train.parse_train_config(["--dropout_p=0.3"]).dropout_p == 0.3


@pytest.mark.parametrize("policy_type", ["mse", "flow"])
@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("dropout_override", [None, 0.0, 0.3])
def test_evaluation_does_not_change_training(
    tmp_path, policy_type, warm_start, dropout_override
):
    """Evaluation frequency must not alter dropout training or its RNG.

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

    checkpoint = tmp_path / "initial.pt"
    if warm_start:
        initial_policy = make_policy(policy_type)
        initial_policy.model.dropout_0.p = 0.2
        initial_policy.model.dropout_1.p = 0.4
        save_policy(checkpoint, initial_policy, make_normalizer())

    layer_count = 2 if warm_start else 1
    expected_probabilities = (
        [dropout_override] * layer_count
        if dropout_override is not None
        else ([0.2, 0.4] if warm_start else [0.0])
    )

    def run(root: Path, evaluation_interval: int):
        losses: list[float] = []
        real_step = train.train_step
        training_generator = None
        evaluation_steps = []

        def assert_evaluation_preserves_training_state(evaluate):
            def wrapped(*args, **kwargs):
                model = kwargs["model"]
                training_modes_before = [module.training for module in model.modules()]
                rng_state_before = training_generator.get_state().clone()

                def assert_eval_mode(operation):
                    def checked(*args, **kwargs):
                        assert all(not module.training for module in model.modules())
                        return operation(*args, **kwargs)

                    return checked

                with (
                    patch.object(
                        model,
                        "compute_loss",
                        side_effect=assert_eval_mode(model.compute_loss),
                    ),
                    patch.object(
                        model,
                        "sample_actions",
                        side_effect=assert_eval_mode(model.sample_actions),
                    ),
                ):
                    result = evaluate(*args, **kwargs)
                assert [
                    module.training for module in model.modules()
                ] == training_modes_before
                assert torch.equal(rng_state_before, training_generator.get_state())
                evaluation_steps.append(len(losses))
                return result

            return wrapped

        def record_train_step(model, *args, **kwargs):
            nonlocal training_generator
            training_generator = kwargs["generator"]
            layers = [
                m for m in model.modules() if isinstance(m, DropoutWithExplicitRNG)
            ]
            assert [m.p for m in layers] == expected_probabilities
            loss = real_step(model, *args, **kwargs)
            losses.append(loss.item())
            return loss

        config = train.TrainConfig(
            policy_type=policy_type,
            hidden_dims=(4,),
            chunk_size=1,
            num_epochs=2,
            batch_size=4,
            validation_interval=evaluation_interval,
            eval_interval=evaluation_interval,
            eval_episodes=2,
            final_eval_episodes=2,
            flow_num_steps=2,
            num_video_episodes=0,
            save_checkpoints=False,
            log_wandb=False,
            log_csv=False,
            show_summary=False,
            dropout_p=dropout_override,
            init_from=checkpoint if warm_start else None,
        )
        with (
            patch.object(train, "LOG_ROOT", str(root)),
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
            patch.object(train, "train_step", side_effect=record_train_step),
            patch.object(
                train,
                "evaluate_policy",
                side_effect=assert_evaluation_preserves_training_state(
                    train.evaluate_policy
                ),
            ),
            patch.object(
                train,
                "compute_validation_loss",
                side_effect=assert_evaluation_preserves_training_state(
                    train.compute_validation_loss
                ),
            ),
        ):
            train.run_training(config)
        assert losses
        if evaluation_interval == 2:
            assert any(step < len(losses) for step in evaluation_steps)
        assert all(np.isfinite(losses))
        return losses

    # Changing evaluation frequency must not change the training trajectory.
    frequent_eval_losses = run(tmp_path / "frequent_eval_losses", evaluation_interval=2)
    end_only_eval_losses = run(
        tmp_path / "end_only_eval_losses", evaluation_interval=100
    )
    assert frequent_eval_losses == end_only_eval_losses


@pytest.mark.parametrize("policy_type", ["mse", "flow"])
def test_checkpoint_restores_dropout_in_eval_mode(policy_type, tmp_path):
    policy = make_policy(policy_type)
    policy.model.dropout_0.p = 0.1
    policy.model.dropout_1.p = 0.3
    path = tmp_path / "policy.pt"
    save_policy(path, policy, make_normalizer())
    payload = torch.load(path, weights_only=True)
    assert payload["training_config"]["dropout"] == {
        "model.dropout_0": 0.1,
        "model.dropout_1": 0.3,
    }
    restored, _, _ = load_policy(path)
    assert get_dropout_config(restored) == get_dropout_config(policy)
    assert all(not module.training for module in restored.modules())
    state = torch.ones(16, 3)
    actions = torch.ones(16, 1, 2)
    restored.train()
    original_generator = torch.Generator().manual_seed(17)
    restored_generator = torch.Generator().manual_seed(17)
    expected = policy.compute_loss(state, actions, generator=original_generator)
    actual = restored.compute_loss(state, actions, generator=restored_generator)
    assert torch.equal(actual, expected)
    assert torch.equal(original_generator.get_state(), restored_generator.get_state())


@pytest.mark.parametrize("p", [-0.1, 1.0, float("nan")])
def test_invalid_checkpoint_dropout_is_rejected(tmp_path, p):
    path = tmp_path / "invalid.pt"
    save_policy(path, make_policy("mse"), make_normalizer())
    payload = torch.load(path, weights_only=True)
    payload["training_config"]["dropout"]["model.dropout_0"] = p
    torch.save(payload, path)
    with pytest.raises(ValueError, match="dropout probability"):
        load_policy(path)


@pytest.mark.parametrize("change", ["missing", "extra", "renamed"])
def test_checkpoint_rejects_dropout_layer_mismatch(tmp_path, change):
    path = tmp_path / "mismatch.pt"
    save_policy(path, make_policy("mse"), make_normalizer())
    payload = torch.load(path, weights_only=True)
    config = payload["training_config"]["dropout"]
    if change in ("missing", "renamed"):
        del config["model.dropout_0"]
    if change in ("extra", "renamed"):
        config["model.unknown_dropout"] = 0.2
    torch.save(payload, path)
    with pytest.raises(ValueError, match="dropout layer names"):
        load_policy(path)


def test_checkpoint_without_dropout_layers(tmp_path):
    policy = build_policy(
        PolicyConfig("mse", state_dim=3, action_dim=2, chunk_size=1, hidden_dims=()),
        cpu_generator=torch.Generator().manual_seed(42),
    )
    override_dropout_p(policy, 0.3)
    path = tmp_path / "linear.pt"
    save_policy(path, policy, make_normalizer())
    restored, _, _ = load_policy(path)
    assert get_dropout_config(restored) == {}
    for name, value in policy.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
