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


class DropoutWithExplicitRNG(ModuleWithExplicitRNG):
    """Inverted dropout with explicit randomness and standard train/eval gating.

    Probability is a runtime setting, like nn.Dropout.p; it is not a parameter
    or buffer. Disabled and evaluation-mode calls consume no random numbers.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p

    def forward(
        self, x: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        if generator is None:
            raise ValueError("Active dropout requires a caller-owned generator")
        # Let PyTorch sample from the scalar probability without rounding it
        # to the activation dtype before deciding which elements survive.
        mask = torch.empty_like(x).bernoulli_(1.0 - self.p, generator=generator)
        # Scale the mask first so backward does not overflow before masking
        # dropped gradients (inf * 0 would produce NaN).
        mask.div_(1.0 - self.p)
        return x * mask

    def extra_repr(self) -> str:
        return f"p={self.p}"
