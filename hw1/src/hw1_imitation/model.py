"""Build and train state-conditioned policies that predict action chunks.

**Policy workflow**

1. Describe architecture with [PolicyConfig][hw1_imitation.model.PolicyConfig].
2. Use [build_policy][hw1_imitation.model.build_policy] to initialize with a
   caller-owned CPU generator and place the policy on `target_device`.
3. Train with `compute_loss`; call `eval()` before inference with `sample_actions`.
   Supply a generator on the sampling device whenever randomness is needed.

[MSEPolicy][hw1_imitation.model.MSEPolicy] predicts actions directly.
[FlowMatchingPolicy][hw1_imitation.model.FlowMatchingPolicy] learns a velocity
field and integrates from Gaussian noise. Both return chunks shaped
`(batch, chunk_size, action_dim)`; normalization belongs to the caller.

Architecture, parameter state, runtime dropout settings, and inference step
counts are separate concerns. Policies do not perform checkpoint I/O.
"""

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
    """Store the architecture needed to reconstruct a policy.

    This frozen dataclass records values without validating them. Constructors
    check their policy kind and delegate layer-size checks to the MLP; this is not
    a comprehensive configuration validator.

    Learning rate, runtime dropout probabilities, and flow integration steps belong
    outside this configuration. Parameters are stored separately in `state_dict()`.

    Attributes:
        policy_type: Policy family: `"mse"` for direct prediction or `"flow"` for
            flow matching. The factory rejects unknown values.
        state_dim: Number of state features expected per observation.
        action_dim: Number of features in one action.
        chunk_size: Prediction horizon: actions produced per policy prediction.
        hidden_dims: Hidden-layer widths in order. An empty tuple produces a single
            linear layer with no hidden activations or dropout.
    """

    policy_type: PolicyType
    state_dim: int
    action_dim: int
    chunk_size: int
    hidden_dims: tuple[int, ...] = (128, 128)


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Define the shared interface for state-conditioned action-chunk policies.

    Subclasses implement initialization, loss computation, and action sampling.
    Use the explicit methods rather than calling the policy as a module: this base
    class does not implement `forward`.

    **State and randomness**

    - Architecture lives in `config`; learned parameters use the inherited
      [PyTorch Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
      state interface.
    - Loss and sampling methods receive caller-owned generators. Active dropout
      and stochastic policy operations consume draws; deterministic paths do not.
    - The caller owns normalization, device placement, and train/eval mode.

    Attributes:
        config: Supplied architecture object, retained by reference.
    """

    def __init__(self, config: PolicyConfig) -> None:
        """Initialize PyTorch module state and retain the architecture.

        Args:
            config: Architecture to expose through the policy's dimension properties.
                This base constructor performs no configuration validation.
        """
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def reset_parameters(self, generator: torch.Generator) -> None:
        """Initialize policy parameters using caller-owned randomness.

        Args:
            generator: Generator whose state advances as parameters are initialized.
                It must match the parameter device; `build_policy` initializes on CPU.

        Subclasses must define initialization for their architecture.
        """

    @abc.abstractmethod
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Compute a scalar training loss with caller-owned randomness.

        Args:
            state: State batch with intended shape `(batch, state_dim)`.
            action_chunk: Target actions with intended shape
                `(batch, chunk_size, action_dim)`, in the caller's training coordinates.
            generator: Generator on the sampling device. Stochastic loss sampling and
                active dropout advance it; otherwise it remains untouched.

        Returns:
            loss: Scalar mean training loss. Concrete subclasses define the target
                and validation behavior.
        """

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        generator: torch.Generator,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Predict an action chunk for each state in a batch.

        Call `eval()` to disable dropout during inference. This interface does not
        promise that every implementation disables gradient recording.

        Args:
            state: State batch with intended shape `(batch, state_dim)`.
            generator: Caller-owned generator on the sampling device. Stochastic
                policies and active dropout consume its state.
            num_steps: Flow integration count; ignored by MSE policies.

        Returns:
            actions: Tensor shaped `(batch, chunk_size, action_dim)` in the policy's
                training coordinates; the caller handles denormalization.
        """

    @property
    def state_dim(self) -> int:
        """Return the configured number of state features.

        Returns:
            count: The architecture's `state_dim` value, without additional validation.
        """
        return self.config.state_dim

    @property
    def action_dim(self) -> int:
        """Return the configured number of features per action.

        Returns:
            count: The architecture's `action_dim` value, without additional validation.
        """
        return self.config.action_dim

    @property
    def chunk_size(self) -> int:
        """Return the configured number of actions per predicted chunk.

        Returns:
            count: The architecture's `chunk_size` value, without additional validation.
        """
        return self.config.chunk_size


class MLP(SequentialWithExplicitRNG):
    """Apply linear layers with ReLU and explicit-RNG dropout between them.

    Every hidden layer is followed by ReLU and dropout, initially with probability
    zero. The output layer is linear. With no hidden layers, the network is a
    single linear layer.

    **Module behavior**

    - Inherit generator routing from
      [SequentialWithExplicitRNG][hw1_imitation.rng_modules.SequentialWithExplicitRNG].
    - Keep numeric linear/activation names stable independently of dropout names;
      dropout has no learned parameters or buffers.
    - Direct construction uses PyTorch's usual layer initialization. Use
      [build_policy][hw1_imitation.model.build_policy] for isolated initialization
      of a complete policy.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
    ) -> None:
        """Construct the ordered linear, activation, and dropout layers.

        Args:
            input_dim: Input feature count.
            hidden_dims: Hidden widths in order; empty means no hidden layers.
            output_dim: Output feature count.

        Raises:
            ValueError: Any input, hidden, or output dimension is nonpositive.

        Integer types and other layer constraints are delegated to PyTorch.
        """
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
        """Reinitialize linear parameters from explicit uniform draws.

        For each linear layer, sample weights and bias independently from
        `[-1/sqrt(in_features), 1/sqrt(in_features)]`, matching its default distribution.
        ReLU and dropout layers are left unchanged, including dropout probabilities.

        Args:
            generator: Generator matching the parameter device. Its state advances
                with each parameter draw; parameter tensors are modified in place.

        Raises:
            TypeError: A child is not a linear, ReLU, or explicit-RNG dropout layer.
                Earlier children may already have been reinitialized.
        """
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
    """Predict action chunks directly and fit them with mean squared error.

    The MLP maps a state to a flattened action chunk. Sampling reshapes that output;
    there is no integration loop or initial-noise draw.

    **Runtime behavior**

    - Active training-mode dropout is the only consumer of the supplied generator.
    - `sample_actions` ignores `num_steps` and does not disable gradient recording.
    - Input compatibility is delegated to tensor operations; targets may broadcast
      under PyTorch's MSE rules rather than receiving strict shape validation.

    Attributes:
        model: Registered MLP mapping state features to `chunk_size * action_dim`
            outputs. The inherited `config` holds its architecture.
    """

    def __init__(self, config: PolicyConfig) -> None:
        """Construct the direct-prediction MLP for an MSE architecture.

        Args:
            config: Architecture with `policy_type="mse"`.

        Raises:
            ValueError: The policy kind is not `"mse"` or an MLP dimension is nonpositive.

        For isolated parameter initialization, use
        [build_policy][hw1_imitation.model.build_policy].
        """
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
        """Compute mean squared error against flattened target actions.

        Args:
            state: State batch, intended shape `(batch, state_dim)`.
            action_chunk: Targets, intended shape `(batch, chunk_size, action_dim)`.
                Flatten all dimensions after the batch axis before comparison. PyTorch
                may broadcast a mismatched target; shapes are not checked explicitly.
            generator: Generator used only by active training-mode dropout. It must
                match that sampling device when dropout is active.

        Returns:
            loss: Scalar MSE averaged over all compared batch and action elements,
                with a gradient graph when gradient recording is enabled.
        """
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
        """Reshape direct MLP predictions into action chunks.

        Call `eval()` to disable dropout. This method does not enter `no_grad` mode.

        Args:
            state: State batch, intended shape `(batch, state_dim)`. Shapes are not
                explicitly validated; linear operations and reshaping determine errors.
            generator: Generator consumed only by active training-mode dropout.
            num_steps: Ignored, including its value and type.

        Returns:
            actions: Predictions shaped `(batch, chunk_size, action_dim)` for an ordinary
                two-dimensional input batch. Values are not denormalized.
        """
        predicted_actions = self.model(state, generator=generator)
        return predicted_actions.reshape(
            -1,
            self.chunk_size,
            self.action_dim,
        )

    def reset_parameters(self, generator: torch.Generator) -> None:
        """Reinitialize the policy MLP without changing its architecture or dropout.

        Args:
            generator: Generator matching the parameter device. Advances as weights
                and biases are overwritten with the MLP's uniform initialization.
        """
        self.model.reset_parameters(generator=generator)


class FlowMatchingPolicy(BasePolicy):
    """Generate action chunks by integrating a learned velocity field.

    **Training and sampling**

    - Train on straight interpolations between Gaussian noise and target chunks.
      The velocity target is the target chunk minus its sampled noise.
    - Sample from Gaussian noise using fixed-step forward Euler integration over
      `[0, 1]`; `num_steps` is an inference setting, not architecture.
    - Both paths consume caller-owned randomness even in evaluation mode. Active
      dropout consumes additional draws; call `eval()` for inference without it.
    - Sampling disables gradient recording; loss computation remains differentiable.

    Attributes:
        model: Registered MLP taking concatenated state, flattened action chunk,
            and scalar time, and predicting a flattened chunk velocity. Architecture
            is retained in the inherited `config`.
    """

    def __init__(self, config: PolicyConfig) -> None:
        """Construct the velocity MLP for a flow-matching architecture.

        Args:
            config: Architecture with `policy_type="flow"`. The input width combines
                state features, flattened chunk features, and one time coordinate.

        Raises:
            ValueError: The policy kind is not `"flow"` or an MLP dimension is nonpositive.

        For isolated parameter initialization, use
        [build_policy][hw1_imitation.model.build_policy].
        """
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
        """Evaluate the conditional velocity field (implementation detail).

        Concatenate inputs along feature axis 1, then apply the MLP. Compatibility is
        delegated to concatenation and linear operations; no input is modified.

        Args:
            state: States shaped `(batch, state_dim)`.
            interpolated_action_chunk: Flattened chunks shaped
                `(batch, chunk_size * action_dim)`.
            tau: Time coordinates shaped `(batch, 1)`.
            generator: Optional generator routed to dropout. Required on the matching
                device if training-mode dropout is active; otherwise unused.

        Returns:
            velocity: Predicted flattened velocities shaped
                `(batch, chunk_size * action_dim)`.

        Raises:
            ValueError: Active dropout requires a generator but none was supplied.
        """
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
        """Fit the velocity field on random noise-to-target interpolations.

        1. Flatten targets and sample matching standard Gaussian noise.
        2. Sample one uniform time in `[0, 1)` per batch item.
        3. Regress the velocity at the interpolated chunk against `target - noise`.

        Args:
            state: State batch, intended shape `(batch, state_dim)`, compatible with
                targets and model parameters in device and dtype.
            action_chunk: Targets, intended shape `(batch, chunk_size, action_dim)`.
                Their device and dtype determine the noise and time samples. Shape
                compatibility is delegated to tensor operations.
            generator: Generator matching `action_chunk.device`. Advances for noise,
                time samples, and any active training-mode dropout.

        Returns:
            loss: Scalar MSE averaged over all batch and flattened action elements.
                The target arrays are not modified.
        """
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
        """Integrate Gaussian noise into action chunks using forward Euler updates.

        Use `num_steps` equal steps of size `1 / num_steps`. Evaluate the velocity at
        left endpoints `0, 1 / num_steps, ...` and finish at time 1. Gradient recording
        is disabled, but train/eval mode is not changed.

        Args:
            state: Two-dimensional states shaped `(batch, state_dim)`. Device and dtype
                determine the initial noise and time tensors and must suit the model.
            generator: Generator matching `state.device`. Advances for initial noise
                and any active dropout. Reuse it across action chunks within an episode.
            num_steps: Positive Python integer controlling integration resolution;
                booleans are rejected. Call `eval()` to disable dropout at each step.

        Returns:
            actions: Tensor shaped `(batch, chunk_size, action_dim)` without a gradient
                graph. The caller handles any action denormalization.

        Raises:
            ValueError: State rank or feature count is wrong, or `num_steps` is nonpositive.
            TypeError: `num_steps` is not a Python integer or is a boolean.
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
        """Reinitialize the policy MLP without changing its architecture or dropout.

        Args:
            generator: Generator matching the parameter device. Advances as weights
                and biases are overwritten with the MLP's uniform initialization.
        """
        self.model.reset_parameters(generator=generator)


def construct_policy(config: PolicyConfig) -> BasePolicy:
    """Construct the policy selected by an architecture configuration.

    This low-level factory uses the current PyTorch device context and ordinary
    layer initialization. It does not isolate default RNG state. Prefer
    [build_policy][hw1_imitation.model.build_policy] for explicit-RNG initialization;
    checkpoint restoration can construct on the meta device before loading weights.

    Args:
        config: Architecture choosing the MSE or flow policy family.

    Returns:
        policy: New policy in its default training mode, retaining `config`.

    Raises:
        ValueError: Policy kind is unknown or a selected MLP dimension is nonpositive.
    """
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
    """Initialize a policy using caller-owned randomness and place it on a device.

    1. Construct layers on the meta device without ordinary parameter storage.
    2. Allocate empty CPU parameters and initialize them from `cpu_generator`.
    3. Move the initialized policy to `target_device`.

    Initialization advances only the supplied generator, leaving default RNG states
    unchanged. Concurrent reproducible calls must use separate generator objects.
    Runtime dropout starts disabled (probability zero); the returned module remains
    in training mode. Device allocation and movement errors propagate from
    PyTorch without being translated into a single exception type.

    Args:
        config: Policy architecture, passed to `construct_policy`.
        cpu_generator: CPU generator for initialization. Its state is consumed;
            no reseeding or state restoration is performed.
        target_device: Destination accepted by PyTorch, such as `"cpu"` or `"cuda"`.

    Returns:
        policy: Initialized policy on the requested device, retaining `config`.

    Raises:
        ValueError: The generator is not on CPU, the policy kind is unknown, or an
            MLP dimension is nonpositive.
    """
    if cpu_generator.device != torch.device("cpu"):
        raise ValueError("Policy initialization requires a CPU generator")

    with torch.device("meta"):
        policy = construct_policy(config)

    policy.to_empty(device="cpu")
    policy.reset_parameters(generator=cpu_generator)
    return policy.to(device=target_device)
