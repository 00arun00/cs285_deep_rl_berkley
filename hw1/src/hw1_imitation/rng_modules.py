"""Explicit routing of caller-owned randomness through neural network layers."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class ModuleWithExplicitRNG(nn.Module, ABC):
    """A module that explicitly accepts a caller-owned generator.

    A missing generator is allowed only when the operation needs no randomness.
    Implementations must never silently fall back to the global RNG.
    """

    @abstractmethod
    def forward(
        self, x: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        raise NotImplementedError


class SequentialWithExplicitRNG(nn.Sequential, ModuleWithExplicitRNG):
    """Route one generator through ModuleWithExplicitRNG children in order.

    Ordinary modules receive only the tensor. Nested containers follow the
    same protocol. Draws share the generator in execution order.
    """

    def forward(
        self, x: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        for module in self:
            if isinstance(module, ModuleWithExplicitRNG):
                x = module(x, generator=generator)
            else:
                x = module(x)
        return x
