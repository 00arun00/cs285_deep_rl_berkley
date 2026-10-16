"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import nn

from hw1_imitation.rng_modules import DropoutWithExplicitRNG, SequentialWithExplicitRNG

PolicyType: TypeAlias = Literal["mse", "flow"]


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
    ) -> torch.Tensor:
        """Compute loss using caller-owned randomness.

        Stochastic loss sampling and active dropout advance generator.
        Otherwise it is left untouched. The generator must match the sampling device.
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
        Call eval() for inference without dropout.
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


class MLP(SequentialWithExplicitRNG):
    """Multilayer perceptron with dropout after hidden activations"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
    ) -> None:
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        if any(dim <= 0 for dim in dims):
            raise ValueError(f"All MLP dimensions must be positive; got {dims}")

        # Keep weight names independent of dropout placement.
        # Dropout has no parameters or buffers.
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for index, (input_size, output_size) in enumerate(zip(dims, dims[1:])):
            layers[str(2 * index)] = nn.Linear(input_size, output_size)
            if index < len(hidden_dims):
                layers[str(2 * index + 1)] = nn.ReLU()
                layers[f"dropout_{index}"] = DropoutWithExplicitRNG()

        super().__init__(layers)

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
            elif isinstance(layer, (nn.ReLU, DropoutWithExplicitRNG)):
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

        self.model = MLP(
            input_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.action_dim * config.chunk_size,
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        generator: torch.Generator,  # used by dropout only in training mode.
    ) -> torch.Tensor:
        target = action_chunk.flatten(start_dim=1)
        return nn.functional.mse_loss(
            input=self.model(state, generator=generator),
            target=target,
        )

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        generator: torch.Generator,  # used by dropout only in training mode.
        num_steps: int = 10,  # not used by the model.
    ) -> torch.Tensor:
        predicted_actions = self.model(state, generator=generator)
        return predicted_actions.reshape(
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

        self.model = MLP(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=output_dim,
        )

    def _predict_velocity(
        self,
        state: torch.Tensor,
        interpolated_action_chunk: torch.Tensor,
        tau: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        model_input = torch.concat(
            tensors=(state, interpolated_action_chunk, tau),
            dim=1,
        )
        return self.model(model_input, generator=generator)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        generator: torch.Generator,
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

        predicted_velocity = self._predict_velocity(
            state=state,
            interpolated_action_chunk=interpolated_action_chunk,
            tau=tau_sample,
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
            predicted_velocity = self._predict_velocity(
                state=state,
                interpolated_action_chunk=interpolated_action_chunk,
                tau=tau,
                generator=generator,
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
