"""Training configuration shared by the Tyro CLI and composed configuration inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import tyro
from pydantic import TypeAdapter

from hw1_imitation.model import PolicyType
from hw1_imitation.randomness import RandomStreamFactory, StreamId


@dataclass(frozen=True, slots=True)
class TrainConfig:
    # Policy initialization.
    # A loaded policy supplies its own architecture and normalizer.
    init_from: Path | None = None

    # Model architecture — used only when init_from is None.
    policy_type: PolicyType = "mse"
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    chunk_size: int = 8

    # Dataset and sample preparation.
    data_dir: Path = Path("data")
    pad_action_chunk_with_last_action: bool = True

    # Training — applies to both fresh and loaded policies.
    num_epochs: int = 400
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0

    # Validation and rollout evaluation, measured in training steps.
    validation_interval: int = 100
    eval_interval: int = 10_000
    eval_episodes: int = 100
    final_eval_episodes: int = 100

    # Retain the best K rollout checkpoints, plus latest (0 = latest only).
    save_checkpoints: bool = True
    checkpoint_top_k: int = 3

    # Rollout inference and video recording.
    # flow_num_steps is ignored by MSE policies.
    flow_num_steps: int = 10
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)

    # Root seed identifies the experiment family.
    # Keep it fixed when varying individual sources of randomness.
    seed: int = 42

    # Independently selectable realizations of each stream.
    data_split_variation: int = 0
    model_init_variation: int = 0
    train_loader_variation: int = 0
    train_loss_variation: int = 0
    validation_loader_variation: int = 0
    validation_loss_variation: int = 0
    eval_env_variation: int = 0
    eval_policy_variation: int = 0

    # Training metrics and experiment tracking.
    log_csv: bool = True
    log_wandb: bool = True
    log_interval: int = 100
    wandb_project: str = "cs285-hw1-imitation-learning"
    exp_name: str | None = None

    # Print the effective configuration and dataset/model facts before training.
    show_summary: bool = True

    def __post_init__(self) -> None:
        for name in (
            "num_epochs",
            "batch_size",
            "chunk_size",
            "log_interval",
            "validation_interval",
            "eval_interval",
            "eval_episodes",
            "final_eval_episodes",
            "flow_num_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        streams = RandomStreamFactory(root_seed=self.seed)
        for stream, variation in (
            (StreamId.DATA_SPLIT, self.data_split_variation),
            (StreamId.MODEL_INIT, self.model_init_variation),
            (StreamId.TRAIN_LOADER, self.train_loader_variation),
            (StreamId.TRAIN_LOSS, self.train_loss_variation),
            (StreamId.VALIDATION_LOADER, self.validation_loader_variation),
            (StreamId.VALIDATION_LOSS, self.validation_loss_variation),
            (StreamId.EVAL_ENV, self.eval_env_variation),
            (StreamId.EVAL_POLICY, self.eval_policy_variation),
        ):
            streams.validate(stream, variation=variation)

        if self.checkpoint_top_k < 0:
            raise ValueError("checkpoint_top_k must be nonnegative")

        if self.num_video_episodes < 0:
            raise ValueError("num_video_episodes must be nonnegative")

        if self.num_video_episodes > 0:
            width, height = self.video_size
            if width <= 0 or height <= 0 or width % 2 or height % 2:
                raise ValueError("Video dimensions must be positive and even")


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


_TRAIN_CONFIG_ADAPTER = TypeAdapter(TrainConfig)
_TRAIN_CONFIG_FIELDS = {field.name for field in fields(TrainConfig)}
_TRAIN_CONFIG_TYPES = get_type_hints(TrainConfig)


def train_config_from_mapping(values: Mapping[str, object]) -> TrainConfig:
    """Validate resolved configuration values and construct the existing dataclass.

    Paths remain relative to the caller's working directory. Lists become tuples,
    optional values and policy literals follow TrainConfig's annotations, and the
    existing __post_init__ checks still apply. This boundary does not read data or
    checkpoints; loaded-policy architecture is resolved by the trainer as before.
    """
    raw = dict(values)
    unknown = raw.keys() - _TRAIN_CONFIG_FIELDS
    if unknown:
        raise ValueError(f"Unknown training fields: {sorted(unknown)}")

    # Pydantic normally accepts True as 1. A configuration flag must never become
    # a count, seed, layer width, or optimizer value by implicit conversion.
    for name, value in raw.items():
        annotation = _TRAIN_CONFIG_TYPES[name]
        if annotation in (int, float):
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric, excluding bool")
        elif get_origin(annotation) is tuple and get_args(annotation)[0] is int:
            if isinstance(value, (list, tuple)) and any(
                isinstance(item, bool) for item in value
            ):
                raise ValueError(f"{name} must contain integers, excluding bool")

    config = _TRAIN_CONFIG_ADAPTER.validate_python(raw)
    for name in ("lr", "weight_decay"):
        value = getattr(config, name)
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    return config
