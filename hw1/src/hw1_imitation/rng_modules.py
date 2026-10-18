"""Route caller-owned randomness through neural network layers.

**Generator flow**

1. Implement [ModuleWithExplicitRNG][hw1_imitation.rng_modules.ModuleWithExplicitRNG]
   for layers that accept a generator.
2. Combine layers with
   [SequentialWithExplicitRNG][hw1_imitation.rng_modules.SequentialWithExplicitRNG]
   to pass one generator through participating children in execution order.
3. Use [DropoutWithExplicitRNG][hw1_imitation.rng_modules.DropoutWithExplicitRNG]
   for dropout that draws from that generator rather than the default RNG.

Dropout probabilities are runtime attributes, outside PyTorch's parameter and
buffer state. [get_dropout_config][hw1_imitation.rng_modules.get_dropout_config]
and [restore_dropout_config][hw1_imitation.rng_modules.restore_dropout_config]
handle their separate checkpoint representation;
[override_dropout_p][hw1_imitation.rng_modules.override_dropout_p] applies one
probability to every matching layer.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class ModuleWithExplicitRNG(nn.Module, ABC):
    """Define a PyTorch module interface with an explicit optional generator.

    A missing generator is allowed only when an operation needs no randomness.
    Implementations must reject missing randomness rather than silently falling
    back to the global RNG. This is a subclass contract, not automatic enforcement
    by the base class.

    Other module behavior comes from
    [PyTorch Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html).
    """

    @abstractmethod
    def forward(
        self, x: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Transform a tensor using caller-owned randomness when needed.

        Args:
            x: Input tensor; concrete layers define shape and dtype requirements.
            generator: Optional caller-owned generator. A stochastic implementation
                must require a compatible generator when its random operation is active.

        Returns:
            result: Transformed tensor defined by the concrete layer.

        Raises:
            NotImplementedError: This abstract base implementation is called directly.
        """
        raise NotImplementedError


class SequentialWithExplicitRNG(nn.Sequential, ModuleWithExplicitRNG):
    """Apply children in order, routing one generator to participating modules.

    **Routing rules**

    - Children implementing `ModuleWithExplicitRNG` receive the same generator.
      Nested explicit-RNG containers follow this protocol recursively.
    - Ordinary children receive only the tensor. An ordinary stochastic layer can
      still use its own default RNG; the container does not make it deterministic.
    - Random draws share the supplied generator in execution order. No generator
      is cloned, reseeded, or restored.

    Construction, child registration, and indexing follow
    [PyTorch Sequential](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html).
    """

    def forward(
        self, x: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Pass the tensor through children while routing the generator selectively.

        Args:
            x: Initial tensor; each child receives the previous child's output.
            generator: Generator passed only to explicit-RNG children. `None` is valid
                if those children do not need randomness; children enforce that rule.

        Returns:
            result: Final child's output, or the original `x` object when empty.
                Shape, dtype, mutation, and failure behavior depend on the children.
        """
        for module in self:
            if isinstance(module, ModuleWithExplicitRNG):
                x = module(x, generator=generator)
            else:
                x = module(x)
        return x


class DropoutWithExplicitRNG(ModuleWithExplicitRNG):
    """Apply inverted dropout with explicit randomness and train/eval gating.

    **Forward behavior**

    - In evaluation mode or with `p == 0`, return the original input object and
      consume no randomness.
    - Otherwise, sample an elementwise Bernoulli mask with keep probability `1-p`,
      scale it by `1/(1-p)`, and multiply without modifying the input.
    - Active dropout requires a generator matching the sampling device. PyTorch
      handles tensor dtype and device compatibility.

    Use inherited `train()` and `eval()` methods to control mode. Direct assignments
    to `p` are not revalidated by `forward`; use the configuration helpers for
    validated updates.

    Attributes:
        p: Drop probability in `[0, 1)`, defaulting to zero. A mutable runtime attribute,
            not a parameter or buffer, so it is absent from `state_dict()`.
    """

    def __init__(self, p: float = 0.0) -> None:
        """Initialize the module and validate its drop probability.

        Args:
            p: Probability satisfying `0 <= p < 1`. The constructor performs a range
                comparison, not a separate numeric-type check.

        Raises:
            ValueError: The probability is outside the range, including NaN and infinities.
            TypeError: The supplied value cannot be compared with numeric bounds.
        """
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p

    def forward(
        self, x: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Return the input unchanged when inactive, otherwise apply a sampled mask.

        Args:
            x: Tensor to mask elementwise. Active sampling uses its shape, dtype and
                device; supported dtypes are determined by PyTorch's mask operations.
            generator: Generator on the sampling device, required only for active
                training-mode dropout. Its state advances when the mask is sampled.

        Returns:
            result: The same `x` object when inactive, or a new masked tensor with the
                same shape when active. Active dropout leaves `x` unchanged and remains
                differentiable with respect to it when gradient recording is enabled.

        Raises:
            ValueError: Training-mode dropout has nonzero probability but no generator.
        """
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
        """Describe the current drop probability in the module representation.

        Returns:
            description: Text of the form `p=<current probability>`.
        """
        return f"p={self.p}"


def get_dropout_config(model: nn.Module) -> dict[str, float]:
    """Snapshot explicit-RNG dropout probabilities by registered module name.

    Use the result alongside the parameter state when writing a checkpoint. Only
    `DropoutWithExplicitRNG` instances are included, not ordinary PyTorch dropout.
    The function reads current values without validating them.

    Args:
        model: Module tree to inspect through `named_modules()`. A matching root
            uses the empty name; shared module instances follow PyTorch's default
            duplicate removal.

    Returns:
        config: New dictionary mapping registered paths to current probabilities.
            Empty when no matching layers exist; changing dictionary entries does
            not update the model.
    """
    return {
        name: module.p
        for name, module in model.named_modules()
        if isinstance(module, DropoutWithExplicitRNG)
    }


def restore_dropout_config(model: nn.Module, config: dict[str, float]) -> None:
    """Validate a complete dropout configuration before updating any layer.

    **Restore rules**

    - Require a dictionary with exactly the current explicit-RNG dropout names.
    - Require every value to be a Python `int` or `float` in `[0, 1)`. Python's
      boolean subtype means `False` is accepted as zero; `True` fails the range.
    - Apply assignments only after all names and values have passed validation.
      Do not change module mode or learned parameters.

    Args:
        model: Module tree whose explicit-RNG dropout layers will be updated.
        config: Full name-to-probability mapping, such as `get_dropout_config`
            returns. An empty mapping is required when there are no matching layers.

    Raises:
        ValueError: Input is not a dictionary, names do not match, or any value has
            an unsupported type or fails the probability range check.
    """
    layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, DropoutWithExplicitRNG)
    }
    if not isinstance(config, dict) or config.keys() != layers.keys():
        raise ValueError("Configured dropout layer names do not match the model")
    for p in config.values():
        if not isinstance(p, (int, float)) or not 0.0 <= p < 1.0:
            raise ValueError("Configured dropout probability must be in [0, 1)")
    for name, p in config.items():
        layers[name].p = p


def override_dropout_p(model: nn.Module, p: float) -> None:
    """Set one validated probability on every explicit-RNG dropout layer.

    Validate before traversal, even when the model contains no matching layers.
    Ordinary PyTorch dropout, training mode, and learned parameters are unchanged.

    Args:
        model: Module tree to traverse through `modules()`.
        p: Shared probability satisfying `0 <= p < 1`. This uses a range comparison
            without the separate numeric-type check performed by the restore helper.

    Raises:
        ValueError: The probability is outside the range, including NaN and infinities.
        TypeError: The supplied value cannot be compared with numeric bounds.
    """
    if not 0.0 <= p < 1.0:
        raise ValueError("dropout probability must be in [0, 1)")
    for module in model.modules():
        if isinstance(module, DropoutWithExplicitRNG):
            module.p = p
