"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import nn

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

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    @property
    def chunk_size(self) -> int:
        return self.config.chunk_size

    @abc.abstractmethod
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Return actions shaped (batch, chunk_size, action_dim).

        num_steps controls integration for flow policies and is ignored
        by the MSE policy.
        """


class SimpleMLP(nn.Sequential):
    """Helper module to construct a simple MLP"""

    def __init__(
        self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int
    ) -> None:
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []

        for idx in range(len(dims) - 1):
            layers.append(
                nn.Linear(
                    in_features=dims[idx],
                    out_features=dims[idx + 1],
                )
            )
            if idx < len(dims) - 2:
                layers.append(nn.ReLU())

        super().__init__(*layers)


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
    ) -> torch.Tensor:
        target = action_chunk.flatten(start_dim=1)
        return nn.functional.mse_loss(
            input=self.model(state),
            target=target,
        )

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        predict = self.model(state)
        return predict.reshape(
            -1,
            self.chunk_size,
            self.action_dim,
        )


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
    ) -> torch.Tensor:
        model_input = torch.concat(
            tensors=(state, interpolated_action_chunk, tau),
            dim=1,
        )
        return self.model(model_input)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
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
        )
        tau_sample = torch.rand(
            size=(batch_size, 1),
            device=device,
            dtype=dtype,
        )

        interpolated_action_chunk = (
            tau_sample * flat_action_chunk + (1 - tau_sample) * noise_sample
        )

        model_input = torch.concat(
            tensors=(state, interpolated_action_chunk, tau_sample),
            dim=1,
        )
        target_vector = flat_action_chunk - noise_sample

        return nn.functional.mse_loss(
            input=self.model(model_input),
            target=target_vector,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Integrate from Gaussian noise to an action chunk.

        Uses num_steps Euler updates over time [0, 1].
        The step count is an inference setting, not an architecture field.
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


def build_policy(config: PolicyConfig) -> BasePolicy:
    """Construct a policy from its architecture configuration."""
    if config.policy_type == "mse":
        return MSEPolicy(config)

    if config.policy_type == "flow":
        return FlowMatchingPolicy(config)

    raise ValueError(f"Unknown policy type: {config.policy_type}")
