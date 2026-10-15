"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import nn

PolicyType: TypeAlias = Literal["mse", "flow"]


def functional_dropout(
    x: torch.Tensor,
    *,
    dropout_p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Drop out elements with inverted scaling, drawing from a caller-owned
    generator.

    Matches nn.functional.dropout semantics, but the mask comes from the
    supplied generator instead of the global RNG state. The generator must
    match the device of x.
    """
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0, 1); got {dropout_p}")

    keep = (
        torch.rand(
            size=x.shape,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        >= dropout_p
    )
    return x * keep / (1.0 - dropout_p)


@dataclass(frozen=True)
class PolicyConfig:
    """Architecture needed to reconstruct a policy.

    chunk_size is the prediction horizon: the number of actions produced
    by one policy prediction.

    Training settings and inference settings, such as learning rate and
    flow integration steps, belong outside this configuration.
    """

    policy_type: PolicyType
    state_dim: int
    action_dim: int
    chunk_size: int
    hidden_dims: tuple[int, ...] = (128, 128)


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action-chunking policies.

    Architecture lives in config; learned parameters live in state_dict().
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def reset_parameters(self, generator: torch.Generator) -> None:
        """Enforce that every policy explicitly defines how its specific
        architecture is initialized using the passed generator.
        """

    @abc.abstractmethod
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        generator: torch.Generator,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Compute loss using caller-owned randomness.

        Stochastic policies advance generator. Deterministic policies leave
        it untouched. The generator must match the sampling device.

        dropout_p applies generator-driven dropout to hidden activations in
        training mode only; model.eval() disables it and draws nothing, and
        0.0 always disables it.
        """

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        generator: torch.Generator,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Return actions shaped (batch, chunk_size, action_dim).

        Stochastic policies advance the caller-owned generator.
        num_steps controls flow integration; MSE policies ignore it.
        """

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    @property
    def chunk_size(self) -> int:
        return self.config.chunk_size


class SimpleMLP(nn.Sequential):
    """Helper module to construct a simple MLP"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
    ) -> None:
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        if any(dim <= 0 for dim in dims):
            raise ValueError(f"All MLP dimensions must be positive; got {dims}")

        layers = []
        for input_size, output_size in zip(dims, dims[1:]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())

        # The output layer has no activation.
        layers.pop()

        super().__init__(*layers)

    def forward_with_dropout(
        self,
        x: torch.Tensor,
        *,
        dropout_p: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Forward pass applying dropout after every hidden activation.

        The output layer never receives dropout. Dropout applies only while
        the module is in training mode: eval() disables it and draws nothing,
        matching nn.Dropout conventions. With dropout_p = 0 or in eval mode
        this reduces to plain forward() and draws nothing from the generator.
        The generator must match the device of x.
        """
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0, 1); got {dropout_p}")

        if dropout_p == 0.0 or not self.training:
            return self(x)

        layers = list(self)
        for layer in layers[:-1]:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = functional_dropout(
                    x,
                    dropout_p=dropout_p,
                    generator=generator,
                )
        return layers[-1](x)

    def reset_parameters(
        self,
        *,
        generator: torch.Generator,
    ) -> None:
        for layer in self:
            if isinstance(layer, nn.Linear):
                # Matches nn.Linear's default distribution for positive fan-in.
                bound = layer.in_features**-0.5

                nn.init.uniform_(
                    layer.weight,
                    -bound,
                    bound,
                    generator=generator,
                )
                if layer.bias is not None:
                    nn.init.uniform_(
                        layer.bias,
                        -bound,
                        bound,
                        generator=generator,
                    )
            elif isinstance(layer, nn.ReLU):
                pass
            else:
                raise TypeError(
                    "No explicit initialization defined for "
                    + f"{type(layer).__name__}"
                )


class MSEPolicy(BasePolicy):
    """Predict action chunks directly, trained with MSE loss."""

    def __init__(self, config: PolicyConfig) -> None:
        if config.policy_type != "mse":
            raise ValueError("MSEPolicy requires policy_type='mse'")

        super().__init__(config)

        self.model = SimpleMLP(
            input_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.action_dim * config.chunk_size,
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        generator: torch.Generator,  # advances only in training with dropout_p > 0.
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        target = action_chunk.flatten(start_dim=1)
        prediction = self.model.forward_with_dropout(
            x=state,
            dropout_p=dropout_p,
            generator=generator,
        )
        return nn.functional.mse_loss(
            input=prediction,
            target=target,
        )

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        generator: torch.Generator,  # model is deterministic, generator not used.
        num_steps: int = 10,  # not used by the model.
    ) -> torch.Tensor:
        predict = self.model(state)
        return predict.reshape(
            -1,
            self.chunk_size,
            self.action_dim,
        )

    def reset_parameters(self, generator: torch.Generator) -> None:
        self.model.reset_parameters(generator=generator)


class FlowMatchingPolicy(BasePolicy):
    """Predict action chunks by integrating a learned velocity field."""

    def __init__(self, config: PolicyConfig) -> None:
        if config.policy_type != "flow":
            raise ValueError("FlowMatchingPolicy requires policy_type='flow'")

        super().__init__(config)

        # Input: observation, flattened action chunk, and time.
        input_dim = config.state_dim + config.action_dim * config.chunk_size + 1
        output_dim = config.action_dim * config.chunk_size

        self.model = SimpleMLP(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=output_dim,
        )

    def _forward(
        self,
        state: torch.Tensor,
        interpolated_action_chunk: torch.Tensor,
        tau: torch.Tensor,
        *,
        dropout_p: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        model_input = torch.concat(
            tensors=(state, interpolated_action_chunk, tau),
            dim=1,
        )
        if dropout_p > 0.0 and self.training:
            if generator is None:
                raise ValueError(
                    "dropout_p > 0 requires a generator on the sampling device"
                )
            return self.model.forward_with_dropout(
                x=model_input,
                dropout_p=dropout_p,
                generator=generator,
            )
        return self.model(model_input)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        generator: torch.Generator,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        # Shape: (batch, chunk_size * action_dim).
        flat_action_chunk = action_chunk.flatten(start_dim=1)

        batch_size, flat_action_chunk_dim = flat_action_chunk.shape
        device = action_chunk.device
        dtype = action_chunk.dtype

        noise_sample = torch.randn(
            size=(batch_size, flat_action_chunk_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        tau_sample = torch.rand(
            size=(batch_size, 1),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        interpolated_action_chunk = (
            tau_sample * flat_action_chunk + (1 - tau_sample) * noise_sample
        )

        target_vector = flat_action_chunk - noise_sample

        predicted_velocity = self._forward(
            state=state,
            interpolated_action_chunk=interpolated_action_chunk,
            tau=tau_sample,
            dropout_p=dropout_p,
            generator=generator,
        )

        return nn.functional.mse_loss(
            input=predicted_velocity,
            target=target_vector,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        generator: torch.Generator,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Integrate from Gaussian noise to an action chunk.

        Uses num_steps Euler updates over time [0, 1].
        The step count is an inference setting, not an architecture field.

        The generator must match state.device. The caller should reuse it across
        action chunks within an episode.
        """
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected state shape (batch_size, {self.state_dim}), "
                f"got {tuple(state.shape)}"
            )

        if isinstance(num_steps, bool) or not isinstance(num_steps, int):
            raise TypeError("num_steps must be an integer")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        batch_size = state.shape[0]

        interpolated_action_chunk = torch.randn(
            size=(batch_size, self.action_dim * self.chunk_size),
            device=state.device,
            dtype=state.dtype,
            generator=generator,
        )

        dt = 1 / num_steps
        for step in range(num_steps):
            tau = state.new_full(
                size=(batch_size, 1),
                fill_value=step * dt,
            )
            predicted_velocity = self._forward(
                state=state,
                interpolated_action_chunk=interpolated_action_chunk,
                tau=tau,
            )
            interpolated_action_chunk = (
                interpolated_action_chunk + dt * predicted_velocity
            )

        return interpolated_action_chunk.reshape(
            batch_size,
            self.chunk_size,
            self.action_dim,
        )

    def reset_parameters(self, generator: torch.Generator) -> None:
        self.model.reset_parameters(generator=generator)


def construct_policy(config: PolicyConfig) -> BasePolicy:
    """Construct a policy from its architecture configuration."""
    if config.policy_type == "mse":
        return MSEPolicy(config)

    if config.policy_type == "flow":
        return FlowMatchingPolicy(config)

    raise ValueError(f"Unknown policy type: {config.policy_type}")


def build_policy(
    config: PolicyConfig,
    *,
    cpu_generator: torch.Generator,
    target_device: torch.device | str = "cpu",
) -> BasePolicy:
    """Return an initialized policy on target_device.

    Initialization advances only the caller-owned cpu_generator.
    For reproducible initialization, concurrent calls must use separate
    generator objects.
    Default RNG states are not modified.
    """
    if cpu_generator.device != torch.device("cpu"):
        raise ValueError("Policy initialization requires a CPU generator")

    with torch.device("meta"):
        policy = construct_policy(config)

    policy.to_empty(device="cpu")
    policy.reset_parameters(generator=cpu_generator)
    return policy.to(device=target_device)
