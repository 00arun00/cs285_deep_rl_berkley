"""Immutable measurements and output references with no training dependencies."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvaluationResults:
    """Measurements and completed video paths from one evaluation.

    Paths refer to persistent files owned by the experiment.
    Consumers may read them but do not acquire deletion ownership.

    episode_max_rewards follows evaluation episode-index order.
    """

    mean_reward: float
    num_episodes: int
    video_paths: tuple[Path, ...]
    episode_max_rewards: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Completed run measurements and references to retained local outputs.

    Best follows existing rollout retention: higher scores win, with earlier
    steps winning ties. It is absent when top-K retention is zero. Both checkpoint
    references are absent when saving is disabled. Pending cleanup records are
    excluded because their local files may already have been deleted.
    """

    output_dir: Path
    optimizer_steps: int
    examples_processed: int
    final_evaluation: EvaluationResults
    latest_checkpoint: Path | None
    best_checkpoint: Path | None
    pending_cleanup_steps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrainingEvaluation:
    """Snapshot emitted after rollout logging and optional checkpoint retention.

    The checkpoint exists at notification time; later retention may delete it.
    is_final identifies the last scheduled evaluation, not successful run cleanup.
    """

    optimizer_steps: int
    examples_processed: int
    is_final: bool
    evaluation: EvaluationResults
    checkpoint: Path | None
