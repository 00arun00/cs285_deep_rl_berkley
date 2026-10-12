"""Plateau detection independent of checkpoint selection and experiment tracking."""

from dataclasses import dataclass
import math


@dataclass
class EarlyStopping:
    """Count stale evaluations after a warm-up measured in optimizer steps.

    The first eligible evaluation establishes the reference. Small improvements
    accumulate against that reference until they reach min_delta. Checkpoint
    selection can independently retain any higher score, however small the gain.
    """

    patience: int
    min_delta: float
    start_step: int
    reference_reward: float | None = None
    stale_evals: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.start_step < 0:
            raise ValueError("start_step must be nonnegative")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")

    def update(self, *, step: int, reward: float) -> bool:
        if not math.isfinite(reward):
            raise ValueError("Evaluation reward must be finite")
        if step < self.start_step:
            return False
        if self.reference_reward is None:
            self.reference_reward = reward
            return False

        improvement = reward - self.reference_reward
        if improvement > 0 and improvement >= self.min_delta:
            self.reference_reward = reward
            self.stale_evals = 0
        else:
            self.stale_evals += 1
        return self.stale_evals >= self.patience
