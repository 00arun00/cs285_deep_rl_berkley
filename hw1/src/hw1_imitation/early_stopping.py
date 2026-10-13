"""Plateau detection independent of checkpoint selection and experiment tracking."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

ObjectiveType = Literal["min", "max"]


@dataclass
class EarlyStopping:
    """Count stale evaluations after a warm-up measured in optimizer steps.

    Track a reference value from the first evaluation, including during warm-up.
    Small improvements accumulate against that reference until the gain strictly
    exceeds min_delta in the objective's direction. Such an improvement updates
    the reference and resets patience.

    Evaluations without sufficient improvement count toward patience only when
    step > start_step. Stop after patience consecutive stale evaluations.
    Checkpoint selection independently retains the best observed score,
    however small the improvement.
    """

    patience: int
    min_delta: float
    start_step: int
    objective: ObjectiveType

    stale_steps: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.start_step < 0:
            raise ValueError("start_step must be nonnegative")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        if self.objective not in ("min", "max"):
            raise ValueError(
                f"objective should be 'min' or 'max' but got {self.objective}"
            )

        # First initialization.
        self.reference_reward = np.inf if self.objective == "min" else -np.inf

        if self.objective == "min":
            self._is_better = lambda current: current < (
                self.reference_reward - self.min_delta
            )
        else:
            self._is_better = lambda current: current > (
                self.reference_reward + self.min_delta
            )

    def update(self, value: float, step: int) -> bool:
        if not math.isfinite(value):
            raise ValueError("Value tracked must be finite")

        if self._is_better(value):
            self.reference_reward = value
            self.stale_steps = 0
        else:
            if step > self.start_step:
                self.stale_steps += 1

        return self.stale_steps >= self.patience
