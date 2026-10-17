"""Detect plateaus in an evaluation metric.

Use [EarlyStopping][hw1_imitation.early_stopping.EarlyStopping] to decide when
to stop training. Call
[update][hw1_imitation.early_stopping.EarlyStopping.update] once per evaluation
and let the training loop act on its result.

The training driver maximizes mean evaluation reward. It converts warm-up
epochs into optimizer steps, while patience counts evaluation calls.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

ObjectiveType = Literal["min", "max"]


@dataclass
class EarlyStopping:
    """Request stopping after too many evaluations without sufficient improvement.

    **How it works**

    - The first finite observation establishes a reference, even during warm-up.
    - An improvement must strictly exceed `min_delta` in the chosen direction.
      It replaces the reference and resets the stale-evaluation counter.
    - Otherwise, increment the counter only when `step > start_step`.
      Return `True` when the counter reaches `patience`.

    **Important behavior**

    - Small gains can accumulate against the last reference, not the previous
      value.
    - Patience counts stale evaluations, not optimizer steps or epochs.
    - A stop signal is not latched: a later sufficient improvement clears it.
      The caller decides whether to exit the training loop.
    - Checkpoint selection is separate and can retain a better score even when
      its improvement is too small to reset patience.

    See [update][hw1_imitation.early_stopping.EarlyStopping.update] for a worked
    example.

    Attributes:
        patience: Positive number of stale evaluation updates needed to request
            stopping. This counts calls after warm-up, not elapsed optimizer
            steps or epochs.
        min_delta: Finite, nonnegative improvement threshold. A gain equal to
            this threshold does not reset patience, even when the threshold is
            zero.
        start_step: Nonnegative warm-up boundary in the caller's step units.
            Stale observations at or before this step do not increment the
            counter; sufficient improvements still update the reference.
        objective: Direction selected at construction: `"min"` for lower values
            or `"max"` for higher values. Reassigning this field does not rebuild
            the comparison function.
        stale_steps: Mutable stale-evaluation counter, initially zero unless
            supplied to the constructor. It is not validated or reset by
            initialization; a sufficient improvement resets it to zero.
        reference_reward: Reference observation for measuring improvement,
            despite the reward-specific name. Initialized to positive infinity
            for minimization or negative infinity for maximization, then
            replaced by each sufficiently improved finite observation.
        _is_better: Implementation detail: comparison callable selected during
            initialization. It reads the current `reference_reward` and
            `min_delta` when evaluating an observation.

    """

    patience: int
    min_delta: float
    start_step: int
    objective: ObjectiveType

    stale_steps: int = 0

    def __post_init__(self) -> None:
        """Validate settings and initialize the reference and comparison function.

        Called automatically by the dataclass constructor.

        - Set the reference to positive infinity for `"min"`, or negative
          infinity for `"max"`.
        - Select the comparison direction from `objective`.
        - Leave `stale_steps` as supplied; this is not a full tracker reset.

        Raises:
            ValueError: If `patience` is nonpositive, `start_step` is negative,
                `min_delta` is nonfinite or negative, or `objective` is neither
                `"min"` nor `"max"`.
        """
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
        """Record one evaluation and report whether patience is exhausted.

        Apply the class's improvement and warm-up rules to mutate
        `reference_reward` and `stale_steps`. Reject nonfinite values before
        changing either field.

        Args:
            value: Finite observation to minimize or maximize. Compare against
                the stored reference using the strict `min_delta` threshold.
            step: Progress coordinate used only for comparison with `start_step`.
                Training passes the optimizer step. No monotonicity or
                nonnegativity check is performed; repeated calls at the same
                eligible step each count as an evaluation.

        Returns:
            stop_requested: Whether the current `stale_steps` is at least
                `patience`. This signal is not latched: a later sufficient
                improvement resets the counter and returns `False`.

        Raises:
            ValueError: If `value` is NaN or either infinity.

        Examples:
            Warm up through step 10, then stop after two stale evaluations. A gain
            exactly equal to `min_delta` does not count as an improvement. The
            reference starts at 10, stays unchanged at 12, then advances to 13.
            Observations of 14 and 15 exhaust patience against that reference.

            >>> tracker = EarlyStopping(2, 2.0, 10, objective="max")
            >>> tracker.update(10.0, step=0)
            False
            >>> tracker.update(12.0, step=10)
            False
            >>> tracker.update(13.0, step=11)
            False
            >>> tracker.update(14.0, step=20)
            False
            >>> tracker.update(15.0, step=30)
            True
        """
        if not math.isfinite(value):
            raise ValueError("Value tracked must be finite")

        if self._is_better(value):
            self.reference_reward = value
            self.stale_steps = 0
        else:
            if step > self.start_step:
                self.stale_steps += 1

        return self.stale_steps >= self.patience
