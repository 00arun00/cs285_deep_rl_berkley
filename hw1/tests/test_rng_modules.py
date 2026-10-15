"""RNG-aware composition keeps routing explicit and ordinary layers usable."""

import pytest
import torch
from torch import nn

from hw1_imitation.rng_modules import ModuleWithExplicitRNG, SequentialWithExplicitRNG


class RandomOffset(ModuleWithExplicitRNG):
    def forward(self, x, *, generator=None):
        if generator is None:
            raise ValueError("generator required")
        return x + torch.rand(x.shape, generator=generator)


def test_nested_containers_share_the_generator_in_execution_order():
    network = SequentialWithExplicitRNG(
        RandomOffset(), nn.ReLU(), SequentialWithExplicitRNG(RandomOffset())
    )
    source = torch.tensor([-0.5, 0.5])
    generator = torch.Generator().manual_seed(13)
    reference = torch.Generator().manual_seed(13)
    global_rng_state_before = torch.get_rng_state().clone()
    expected = torch.relu(source + torch.rand(source.shape, generator=reference))
    expected += torch.rand(source.shape, generator=reference)
    torch.testing.assert_close(network(source, generator=generator), expected)
    assert torch.equal(generator.get_state(), reference.get_state())
    assert torch.equal(torch.get_rng_state(), global_rng_state_before)


def test_ordinary_layers_and_empty_container_do_not_require_a_generator():
    x = torch.tensor([-1.0, 2.0], requires_grad=True)
    assert SequentialWithExplicitRNG()(x) is x
    network = SequentialWithExplicitRNG(nn.ReLU(), nn.Identity())
    torch.testing.assert_close(network(x), torch.relu(x))
    network(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.0, 1.0]))


def test_training_mode_propagates_to_nested_rng_modules():
    child = RandomOffset()
    network = SequentialWithExplicitRNG(nn.Identity(), SequentialWithExplicitRNG(child))
    network.eval()
    assert all(not module.training for module in network.modules())
    network.train()
    assert all(module.training for module in network.modules())


def test_missing_generator_error_is_not_swallowed():
    with pytest.raises(ValueError, match="generator required"):
        SequentialWithExplicitRNG(RandomOffset())(torch.ones(2))
