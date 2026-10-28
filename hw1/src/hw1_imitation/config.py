from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import tyro

from hw1_imitation.model import PolicyType
from hw1_imitation.randomness import RandomStreamFactory, StreamId


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Configure a Push-T imitation-learning training run.

    Architecture settings apply only to fresh policies. Training settings also
    apply when initializing from a checkpoint; loading does not resume the
    optimizer or the training-step counter.

    Attributes:
        init_from: Checkpoint used to initialize the policy and normalizer. A loaded
            policy supplies its architecture; optimizer state and training step start
            fresh.
        policy_type: Policy family for a fresh model: mean-squared-error imitation or
            flow matching. Ignored when loading a checkpoint.
        hidden_dims: Hidden-layer widths for a fresh policy. Ignored when loading a
            checkpoint.
        chunk_size: Number of actions predicted per observation by a fresh policy.
            Loaded policies retain their own prediction horizon.
        data_dir: Root directory used to find, download, and extract the Push-T dataset.
        pad_action_chunk_with_last_action: Pad incomplete action chunks with the
            episode’s final action when preparing training and validation samples.
        num_epochs: Maximum number of training epochs. Plateau stopping can end the run
            earlier.
        batch_size: Batch size for training and validation. Training drops the final
            incomplete batch; validation retains it.
        lr: Learning rate passed to the fresh AdamW optimizer.
        weight_decay: Weight-decay coefficient passed to AdamW.
        dropout_p: Override the policy’s dropout probability. None preserves loaded
            dropout settings; a fresh policy uses zero dropout.
        validation_interval: Optimizer steps between complete validation-loss passes.
            Validation also runs on the last scheduled training step.
        eval_interval: Optimizer steps between rollout evaluations used for checkpoint
            selection and plateau stopping. Selection evaluation also runs on the last
            scheduled training step.
        eval_episodes: Number of environment episodes in each checkpoint-selection
            evaluation.
        final_eval_episodes: Number of episodes in the separate final evaluation of the
            best retained checkpoint.
        early_stopping_patience: Consecutive selection evaluations without sufficient
            rollout-reward improvement before stopping, after warm-up. None disables
            plateau stopping.
        early_stopping_min_epochs: Warm-up duration converted to optimizer steps using
            the training-loader length. Stale evaluations count only after that step
            threshold.
        early_stopping_min_delta: Rollout-reward gain that must be strictly exceeded to
            reset patience. Small gains can accumulate against the last reference reward.
        save_checkpoints: Keep checkpoints in the experiment directory. If false,
            checkpoints are still written to a temporary directory, removed after
            success and retained after failure; W&B uploads are controlled separately.
        checkpoint_top_k: Number of best rollout-scored checkpoints retained in addition
            to the latest checkpoint. Zero retains only the latest.
        flow_num_steps: Integration steps for flow-policy rollout sampling, also
            recorded with checkpoints. Ignored by MSE policies.
        num_video_episodes: Maximum episodes recorded per selection or final evaluation.
            Zero disables video recording.
        video_size: Video frame dimensions as width and height. Both must be positive
            and even when recording is enabled.
        seed: Nonnegative root seed defining the experiment’s random-stream family. Keep
            it fixed when varying individual streams.
        data_split_variation: Realization of the stream that selects the fixed training
            and validation episode split.
        model_init_variation: Realization of the parameter-initialization stream for a
            fresh policy. Not used for loaded weights.
        train_loader_variation: Realization of the training-loader generator, which
            advances across epoch shuffles.
        train_loss_variation: Realization of the training-loss generator, including
            stochastic loss sampling and active dropout.
        validation_loader_variation: Realization of the validation-loader generator.
            Validation sample order remains fixed.
        validation_loss_variation: Realization of the loss generator restarted for each
            validation pass to use matching random draws.
        eval_env_variation: Realization of environment randomness for
            checkpoint-selection rollouts.
        eval_policy_variation: Realization of policy-sampling randomness for
            checkpoint-selection rollouts.
        final_eval_env_variation: Realization of the separate final-evaluation
            environment stream.
        final_eval_policy_variation: Realization of the separate final-evaluation
            policy-sampling stream.
        log_csv: Write training, validation, selection-evaluation, and final-evaluation
            CSV metrics.
        log_wandb: Initialize W&B and upload metrics, videos, and checkpoint artifacts.
            False disables W&B integration.
        log_interval: Optimizer steps between training-metric reports. Metrics are also
            reported on the last scheduled training step.
        wandb_project: W&B project used when W&B logging is enabled.
        exp_name: Optional suffix appended to the automatically generated experiment name
            containing the seed and timestamp.
        show_summary: Print effective configuration and dataset/model information before
            training.
    """

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
    # None inherits loaded dropout; fresh models use zero.
    dropout_p: float | None = None

    # Validation and rollout evaluation, measured in training steps.
    validation_interval: int = 100
    eval_interval: int = 10_000
    eval_episodes: int = 100
    final_eval_episodes: int = 100

    # None disables plateau stopping. Warm-up does not count toward patience.
    early_stopping_patience: int | None = None
    early_stopping_min_epochs: int = 100
    early_stopping_min_delta: float = 0.005

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
    final_eval_env_variation: int = 0
    final_eval_policy_variation: int = 0

    # Training metrics and experiment tracking.
    log_csv: bool = True
    log_wandb: bool = True
    log_interval: int = 100
    wandb_project: str = "cs285-hw1-imitation-learning"
    exp_name: str | None = None

    # Print the effective configuration and dataset/model facts before training.
    show_summary: bool = True

    def __post_init__(self) -> None:
        """Check supported configuration bounds when the dataclass is created.

        - Counts and intervals: num_epochs, batch_size, chunk_size, log_interval,
          validation_interval, eval_interval, eval_episodes, final_eval_episodes,
          and flow_num_steps must be positive.
        - Plateau stopping: if set, early_stopping_patience must be positive.
          early_stopping_min_epochs must be nonnegative; early_stopping_min_delta
          must be finite and nonnegative.
        - Dropout: if set, dropout_p must be in [0, 1).
        - Checkpoints and video: checkpoint_top_k and num_video_episodes must be
          nonnegative. With recording enabled, both video dimensions must be
          positive and even.
        - Randomness: the root seed must be a nonnegative integer. Training,
          validation, and selection stream variations must be integers in
          [0, 2**32). Booleans are rejected. Final-evaluation variations are
          validated when their streams are used, rather than here.

        Raises:
            TypeError: The root seed or a checked stream variation is not a
                Python integer (booleans are rejected).
            ValueError: A checked value lies outside its supported bounds.
        """
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

        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_min_epochs < 0:
            raise ValueError("early_stopping_min_epochs must be nonnegative")
        if (
            not math.isfinite(self.early_stopping_min_delta)
            or self.early_stopping_min_delta < 0
        ):
            raise ValueError("early_stopping_min_delta must be finite and nonnegative")

        if self.dropout_p is not None and not 0.0 <= self.dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0, 1)")

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
    """Parse command-line options into a validated training configuration.

    Use Tyro's dataclass interface. When no defaults are supplied, construct the
    standard `TrainConfig` first; parsed construction runs its validation hook.

    Args:
        args: Explicit option strings, or `None` to read process command-line arguments.
        defaults: Default configuration for omitted options, or `None` for standard
            field defaults.
        description: Description displayed by Tyro's generated help.

    Returns:
        config: Parsed training configuration.

    Raises:
        SystemExit: Tyro handles help or reports command-line usage errors.

    Configuration validation errors propagate from `TrainConfig`.
    """
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
        config=(
            tyro.conf.FlagConversionOff,
            tyro.conf.UsePythonSyntaxForLiteralCollections,
        ),
    )


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    """Copy configuration fields into a dictionary with top-level paths as strings.

    Use dataclass `asdict`, then convert values that are directly `Path` instances.
    This is not a general recursive JSON serializer; tuple values remain tuples.
    The original configuration is unchanged.

    Args:
        config: Training configuration to represent for logging.

    Returns:
        values: Dictionary of copied field values, with top-level paths converted
            to strings without resolving them.
    """
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data



