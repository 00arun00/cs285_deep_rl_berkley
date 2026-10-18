"""Save and load self-contained policies for inference or a new training run.

**Checkpoint contents**

- Architecture configuration and learned model weights.
- Runtime training configuration: per-layer dropout probabilities.
- Observation/action normalization statistics and default flow inference steps.

Use [save_policy][hw1_imitation.checkpoint.save_policy] to write a checkpoint,
then [load_policy][hw1_imitation.checkpoint.load_policy] to reconstruct it without
the original dataset. Weights are stored on CPU; configuration is plain data.

Loading does not resume a training session. No optimizer, training position,
or saved RNG state is restored. Further training uses a new optimizer and
caller-owned randomness; saved dropout settings can be explicitly overridden.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from hw1_imitation.data import Normalizer
from hw1_imitation.model import BasePolicy, PolicyConfig, construct_policy
from hw1_imitation.rng_modules import get_dropout_config, restore_dropout_config

CHECKPOINT_VERSION = 1


def save_policy(
    path: Path | str,
    model: BasePolicy,
    normalizer: Normalizer,
    *,
    flow_num_steps: int = 10,
) -> None:
    """Write the policy, normalization statistics, and inference defaults.

    **Write sequence**

    1. Validate inference steps and normalizer dimensions; assemble CPU weights,
       copied normalization tensors, and plain configuration data.
    2. Create missing parent directories and serialize to a sibling `.tmp` file.
    3. Replace the destination after serialization succeeds, then clean up the
       temporary path.

    Calls targeting the same path must not run concurrently because the temporary
    name is shared. Keep model parameters and statistics stable while saving:
    CPU weight tensors are detached but not necessarily copied. This helper does
    not change the model's train/eval mode.

    Filesystem and PyTorch serialization errors propagate to the caller. Replacing
    a file after serialization is not a guarantee of crash-durable storage.

    Args:
        path: Destination checkpoint file. An existing file is replaced after
            successful serialization to `path` with `.tmp` appended to its suffix.
        model: Policy whose architecture, weights, and dropout settings are saved.
        normalizer: Statistics used for training, with dimensions matching the
            policy. Export revalidates statistics and copies their numerical storage.
        flow_num_steps: Positive Python integer default for flow inference. Booleans
            are rejected. Validated even for MSE policies, which ignore it at sampling.

    Raises:
        ValueError: Inference steps are invalid, normalizer dimensions do not match,
            or stored statistics fail validation.
        OSError: A directory, temporary-file, replacement, or cleanup operation fails.
    """
    if (
        isinstance(flow_num_steps, bool)
        or not isinstance(flow_num_steps, int)
        or flow_num_steps <= 0
    ):
        raise ValueError("flow_num_steps must be a positive integer")

    normalizer.validate_dimensions(
        state_dim=model.config.state_dim,
        action_dim=model.config.action_dim,
    )

    checkpoint = {
        "format_version": CHECKPOINT_VERSION,
        "model_config": asdict(model.config),
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "training_config": {"dropout": get_dropout_config(model)},
        "normalizer": normalizer.state_dict(),
        "inference_config": {
            "flow_num_steps": flow_num_steps,
        },
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_policy(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[BasePolicy, Normalizer, dict[str, int]]:
    """Reconstruct a checkpointed policy and return it in evaluation mode.

    **Load sequence**

    1. Read tensors onto CPU with `weights_only=True` and check the format version.
    2. Reconstruct architecture and normalization statistics; validate dimensions
       and the saved positive integer flow-step default.
    3. Construct the model on meta, allocate CPU storage, and strictly load weights.
    4. Restore per-layer dropout, transfer to `device`, and call `eval()`.

    Model construction and loading do not consume default RNG streams. Inference
    randomness remains caller-owned and must be supplied when sampling actions.
    For further training, create a new optimizer and call `model.train()`; use
    [override_dropout_p][hw1_imitation.rng_modules.override_dropout_p] to replace
    saved dropout probabilities if desired.

    The loader checks required components as it uses them rather than performing
    an exhaustive schema check up front. PyTorch deserialization and device errors
    propagate; no device fallback is attempted.

    Args:
        path: Checkpoint produced by [save_policy][hw1_imitation.checkpoint.save_policy].
        device: Destination device for the reconstructed policy. Normalizer arrays
            remain NumPy arrays on CPU.

    Returns:
        model: Loaded policy in evaluation mode on `device`.
        normalizer: Saved state/action statistics with independent copied storage.
        inference_config: Dictionary containing the validated `flow_num_steps` default.

    Raises:
        ValueError: The outer object or format version is unsupported, architecture
            or normalization is invalid, flow steps are invalid, or dropout settings
            do not match the reconstructed model.
        KeyError: A required checkpoint component is missing.
        TypeError: Malformed architecture arguments or unsupported normalization
            tensor conversions fail. Other malformed inputs can also fail in PyTorch.
        RuntimeError: Strict weight loading rejects missing, unexpected, or
            incompatible parameter entries, or a PyTorch operation fails.
        OSError: The checkpoint file cannot be read.
    """
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format_version") != CHECKPOINT_VERSION
    ):
        raise ValueError(
            f"Unsupported checkpoint format; expected version {CHECKPOINT_VERSION}"
        )

    policy_config = PolicyConfig(**checkpoint["model_config"])
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    normalizer.validate_dimensions(
        state_dim=policy_config.state_dim,
        action_dim=policy_config.action_dim,
    )

    flow_num_steps = checkpoint["inference_config"]["flow_num_steps"]
    if (
        isinstance(flow_num_steps, bool)
        or not isinstance(flow_num_steps, int)
        or flow_num_steps <= 0
    ):
        raise ValueError("Checkpoint flow_num_steps must be a positive integer")

    with torch.device("meta"):
        model = construct_policy(policy_config)

    model.to_empty(device="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    restore_dropout_config(model, checkpoint["training_config"]["dropout"])

    model = model.to(device=device)
    model.eval()

    return (
        model,
        normalizer,
        {
            "flow_num_steps": flow_num_steps,
        },
    )
