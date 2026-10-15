"""Dropout layers own probability and consume only explicit randomness."""

import pytest
import torch

from hw1_imitation.rng_modules import DropoutWithExplicitRNG, SequentialWithExplicitRNG


@pytest.mark.parametrize("p", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_probability_is_rejected(p):
    with pytest.raises(ValueError, match="probability"):
        DropoutWithExplicitRNG(p)


@pytest.mark.parametrize("eval_mode", [False, True])
def test_disabled_or_eval_dropout_is_identity_without_draws(eval_mode):
    layer = DropoutWithExplicitRNG(0.5 if eval_mode else 0.0)
    if eval_mode:
        layer.eval()
    x = torch.ones(64, requires_grad=True)
    generator = torch.Generator().manual_seed(3)
    rng_state_before = generator.get_state().clone()
    assert layer(x) is x
    assert layer(x, generator=generator) is x
    assert torch.equal(rng_state_before, generator.get_state())
    layer(x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_mask_scaling_gradients_and_rng_isolation():
    layer = DropoutWithExplicitRNG(0.5)
    x = torch.ones(256, requires_grad=True)
    generator = torch.Generator().manual_seed(7)
    global_rng_state_before = torch.get_rng_state().clone()
    output = layer(x, generator=generator)
    assert set(output.tolist()) == {0.0, 2.0}
    output.sum().backward()
    assert torch.equal(x.grad, output)
    assert torch.equal(global_rng_state_before, torch.get_rng_state())


@pytest.mark.parametrize(("p", "upstream_grad"), [(0.99999, 1.0), (0.5, 40_000.0)])
def test_dropped_float16_gradients_stay_zero_when_scaling_would_overflow(
    p, upstream_grad
):
    x = torch.ones(1_000, dtype=torch.float16, requires_grad=True)
    output = DropoutWithExplicitRNG(p)(x, generator=torch.Generator().manual_seed(42))
    dropped = output == 0
    assert dropped.any()
    assert torch.isfinite(output).all()

    output.backward(torch.full_like(output, upstream_grad))
    assert torch.equal(x.grad[dropped], torch.zeros_like(x.grad[dropped]))


def test_replay_and_advancement_through_container():
    network = SequentialWithExplicitRNG(DropoutWithExplicitRNG(0.5))
    x = torch.ones(256)
    generator = torch.Generator().manual_seed(9)
    first = network(x, generator=generator)
    replay = network(x, generator=torch.Generator().manual_seed(9))
    assert torch.equal(first, replay)
    assert not torch.equal(first, network(x, generator=generator))
    network.eval()
    assert network(x) is x
    network.train()
    with pytest.raises(ValueError, match="generator"):
        network(x)
    assert not torch.equal(network(x, generator=generator), x)


def test_dropout_state_dict_is_empty():
    assert DropoutWithExplicitRNG(0.5).state_dict() == {}


@pytest.mark.parametrize(
    ("dtype", "p"),
    [(torch.bfloat16, 0.999), (torch.float16, 0.9999)],
)
def test_reduced_precision_dropout_preserves_survivors_and_gradients(dtype, p):
    # These valid probabilities round to 1 in the activation dtype. Sampling
    # uniforms in that dtype and comparing against p would drop every element.
    x = torch.ones(200_000, dtype=dtype, requires_grad=True)
    layer = DropoutWithExplicitRNG(p)
    global_rng_state_before = torch.get_rng_state().clone()
    output = layer(x, generator=torch.Generator().manual_seed(42))

    assert output.dtype == dtype
    survivors = output.count_nonzero().item()
    expected_survivors = x.numel() * (1.0 - p)
    assert 0.25 * expected_survivors < survivors < 2.0 * expected_survivors
    torch.testing.assert_close(
        output[output != 0],
        torch.full((survivors,), 1.0 / (1.0 - p), dtype=dtype),
    )
    output.float().sum().backward()
    assert x.grad.dtype == dtype
    assert torch.equal(x.grad, output)
    assert torch.equal(global_rng_state_before, torch.get_rng_state())
