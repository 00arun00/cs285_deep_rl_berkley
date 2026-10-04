"""Save and load self-contained Push-T policies.

A policy checkpoint contains:
- Architecture configuration.
- Learned model weights.
- Observation and action normalization statistics.
- Default inference settings.

Loading does not restore a training session. Further training uses a
new optimizer, training configuration, seed, and logging run.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from hw1_imitation.data import Normalizer
from hw1_imitation.model import BasePolicy, PolicyConfig, build_policy

CHECKPOINT_VERSION = 1


def save_policy(
    path: Path | str,
    model: BasePolicy,
    normalizer: Normalizer,
    *,
    flow_num_steps: int = 10,
) -> None:
    """Save everything needed to reconstruct and use a policy.

    Args:
        path: Destination checkpoint file.
        model: Policy whose architecture and weights will be saved.
        normalizer: Statistics used to prepare its training data.
        flow_num_steps: Default integration steps for flow inference.
            Ignored by MSE policies.

    Weights are saved on CPU so loading does not require the original
    accelerator. Configuration is saved as plain data, not a pickled
    PolicyConfig instance.

    The destination is replaced only after serialization succeeds.
    Calls writing to the same path must not run concurrently.
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
    """Reconstruct a policy without its original training dataset.

    Args:
        path: Checkpoint produced by save_policy().
        device: Device on which to construct the policy.

    Returns:
        model: Loaded policy in evaluation mode.
        normalizer: Saved observation/action normalization statistics.
        inference_config: Saved inference defaults.

    For further training, construct a new optimizer and call model.train().
    No optimizer state, training position, or saved RNG state is restored.
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

    model = build_policy(policy_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return (
        model,
        normalizer,
        {
            "flow_num_steps": flow_num_steps,
        },
    )
