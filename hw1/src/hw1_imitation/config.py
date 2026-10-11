"""Concrete configuration with no training, accelerator, or launcher imports."""

from dataclasses import field
from pathlib import Path
from typing import Annotated, Literal, Mapping

from pydantic import ConfigDict, Field, StrictBool, TypeAdapter
from pydantic.dataclasses import dataclass

PolicyType = Literal["mse", "flow"]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
Variation = Annotated[int, Field(strict=True, ge=0, lt=2**32)]
OptimizerValue = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
_SCHEMA = ConfigDict(extra="forbid", validate_default=True)


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class DataConfig:
    dir: Path = Path("data")
    pad_action_chunk_with_last_action: StrictBool = True


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class ModelConfig:
    """Requested architecture; observation/action dimensions come from the data."""

    policy_type: PolicyType = "mse"
    hidden_dims: tuple[PositiveInt, ...] = (256, 256, 256)
    chunk_size: PositiveInt = 8


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class OptimizationConfig:
    num_epochs: PositiveInt = 400
    batch_size: PositiveInt = 128
    lr: OptimizerValue = 3e-4
    weight_decay: OptimizerValue = 0.0


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class EvaluationConfig:
    validation_interval: PositiveInt = 100
    interval: PositiveInt = 10_000
    episodes: PositiveInt = 100
    final_episodes: PositiveInt = 100
    flow_num_steps: PositiveInt = 10
    num_video_episodes: NonnegativeInt = 5
    video_size: tuple[
        Annotated[int, Field(strict=True)], Annotated[int, Field(strict=True)]
    ] = (256, 256)

    def __post_init__(self) -> None:
        if self.num_video_episodes and any(n <= 0 or n % 2 for n in self.video_size):
            raise ValueError("Video dimensions must be positive and even")


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class RandomnessConfig:
    seed: NonnegativeInt = 42
    data_split_variation: Variation = 0
    model_init_variation: Variation = 0
    train_loader_variation: Variation = 0
    train_loss_variation: Variation = 0
    validation_loader_variation: Variation = 0
    validation_loss_variation: Variation = 0
    eval_env_variation: Variation = 0
    eval_policy_variation: Variation = 0


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class LoggingConfig:
    csv: StrictBool = True
    wandb: StrictBool = True
    interval: PositiveInt = 100
    project: str = "cs285-hw1-imitation-learning"
    show_summary: StrictBool = True


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class CheckpointConfig:
    enabled: StrictBool = True
    top_k: NonnegativeInt = 3


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig | None = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    randomness: RandomnessConfig = field(default_factory=RandomnessConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    warm_start: Path | None = None

    def __post_init__(self) -> None:
        if (self.model is None) == (self.warm_start is None):
            raise ValueError("Choose model settings OR warm_start with model=null")


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class RunConfig:
    output_root: Path = Path("exp")
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None and (
            not self.name
            or self.name in {".", ".."}
            or "/" in self.name
            or "\\" in self.name
        ):
            raise ValueError("Run name must be a single nonempty path component")


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class ExperimentConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    run: RunConfig = field(default_factory=RunConfig)


def train_config_from_mapping(values: Mapping[str, object]) -> TrainConfig:
    return TypeAdapter(TrainConfig).validate_python(values)


def experiment_config_from_mapping(values: Mapping[str, object]) -> ExperimentConfig:
    return TypeAdapter(ExperimentConfig).validate_python(values)


def config_to_dict(config: TrainConfig | ExperimentConfig) -> dict:
    """JSON-safe nested values for composition, tracking, and snapshots."""
    return TypeAdapter(type(config)).dump_python(config, mode="json")
