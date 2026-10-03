"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class SimpleMLP(nn.Sequential):
    """Helper module to construct a simple MLP"""

    def __init__(
        self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int
    ) -> None:
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(in_features=dims[idx], out_features=dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU())
        super().__init__(*layers)


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.model = SimpleMLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim * chunk_size,
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        target = action_chunk.flatten(start_dim=1)
        return nn.functional.mse_loss(input=self.model(state), target=target)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        predict = self.model(state)
        return predict.reshape(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        input_dim = state_dim + action_dim * chunk_size + 1
        output_dim = action_dim * chunk_size
        self.model = SimpleMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
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
        ) # B x [state_dim + chunk_size * action_dim + 1]
        return self.model(model_input)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        flat_action_chunk = action_chunk.flatten(start_dim=1)  # B x chunk_size*2

        batch_size, flat_action_chunk_dim = flat_action_chunk.shape
        device = action_chunk.device
        dtype = action_chunk.dtype
        noise_sample = torch.randn(
            size=(batch_size, flat_action_chunk_dim),
            device=device,
            dtype=dtype,
        )  # B x chunk_size * action_dim

        tau_sample = torch.rand(
            size=(batch_size, 1),
            device=device,
            dtype=dtype,
        )  # B x 1

        interpolated_action_chunk = (
            tau_sample * flat_action_chunk + (1 - tau_sample) * noise_sample
        )  # B x chunk_size * action_dim

        model_input = torch.concat(
            tensors=(state, interpolated_action_chunk, tau_sample),
            dim=1,
        )  # B x [state_dim + chunk_size * action_dim + 1]

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
        """At inference time,
        - we sample initial noise A(t,0) ∼ N (0, I)
        - integrate the

            ODE d(A(t,τ))
                ---------   = vθ(ot, At,τ , τ )
                    dτ
            from: τ = 0 to τ = 1.

        The simplest integration method is Euler integration, which is given by the following update
            rule:
                At,τ+ 1/n = At,τ + 1/n · vθ(ot, At,τ , τ ),
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

        device = state.device
        dtype = state.dtype
        batch_size = state.shape[0]

        sampled_guassian_noise = torch.randn(
            size=(batch_size, self.action_dim * self.chunk_size),
            device=device,
            dtype=dtype,
        )

        interpolated_action_chunk = sampled_guassian_noise
        dt = 1 / num_steps
        for step in range(num_steps):
            # One time value per batch element: (B, 1).
            tau = state.new_full(size=(batch_size, 1), fill_value=step * dt)
            # Predicted velocity: (B, chunk_size * action_dim).
            predicted_velocity = self._forward(
                state=state,
                interpolated_action_chunk=interpolated_action_chunk,
                tau=tau,
            )
            interpolated_action_chunk = (
                interpolated_action_chunk + dt * predicted_velocity
            )

        return interpolated_action_chunk.reshape(
            shape=(batch_size, self.chunk_size, self.action_dim)
        )


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
