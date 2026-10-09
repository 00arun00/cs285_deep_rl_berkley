"""Stable identities for reproducible random streams.

The identity of a random stream is:
    (root_seed, stream_id, variation, index)

Registry values and the seed-material encoding are permanent contracts.
Changing either changes the resulting randomness.

This module controls random-stream assignment. It does not guarantee
identical numerical results across devices or library versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique

import numpy as np
import torch

_UINT32_LIMIT = 1 << 32  # 2**32


@unique
class StreamId(IntEnum):
    """Permanent IDs for distinct randomness responsibilities.

    Rules:
        * Assign IDs explicitly.
        * Preserve an ID when renaming its member.
        * Never reuse a retired ID.
        * Add a new ID when introducing an independent responsibility.

    Declaration order has no effect on seed derivation.
    """

    DATA_SPLIT = 1  # Choose train/validation episode membership
    MODEL_INIT = 2  # Initialize model weights
    TRAIN_LOADER = 3  # Shuffle training samples and isolate loader randomness
    VALIDATION_LOADER = 4  # Isolate validation-loader randomness
    TRAIN_LOSS = 5
    VALIDATION_LOSS = 6
    EVAL_ENV = 7
    EVAL_POLICY = 8

    # NEXT_ID = 9


@dataclass(frozen=True, slots=True)
class RandomStreamFactory:
    """Stateless factory for reproducible generators and integer seeds.

    Identity: (root_seed, stream_id, variation, index)
        root_seed: Common seed from which all stream identities are derived.
        stream_id: Permanent identifier for a randomness responsibility.
        variation: Independently selected realization of that stream (default 0).
        index: Explicit peer identifier within a variation, such as an evaluation
            episode index (default 0). It never increments automatically.

    Every generator call returns a fresh object at its starting state.
    Create it once and retain it to advance a stream; recreate it to replay.
    No global RNG state is modified.

    Reproducibility assumes unchanged encoding, library behavior, device,
    sampling operations, and arguments. Distinct identities can theoretically
    collide, particularly when reduced to integer seeds.

    Invalid argument types raise TypeError. Out-of-range integers raise
    ValueError. No runtime warnings are emitted by this factory.
    """

    root_seed: int

    def __post_init__(self) -> None:
        self._validate_integer(
            self.root_seed,
            name="root_seed",
            upper_bound=None,
        )

    @staticmethod
    def validate(
        stream: StreamId,
        *,
        variation: int = 0,
        index: int = 0,
    ) -> None:
        """Validate stream coordinates without constructing seed material.

        Raises:
            TypeError: A coordinate has an unsupported type.
            ValueError: An integer coordinate is outside [0, 2**32).
        """
        if not isinstance(stream, StreamId):
            raise TypeError("stream must be a StreamId member")

        for name, value in (
            ("stream ID", int(stream)),
            ("variation", variation),
            ("index", index),
        ):
            RandomStreamFactory._validate_integer(
                value,
                name=name,
                upper_bound=_UINT32_LIMIT,
            )

    @staticmethod
    def _validate_integer(
        value: int,
        *,
        name: str,
        upper_bound: int | None = _UINT32_LIMIT,
    ) -> None:
        """Require a nonnegative Python integer without coercion."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a Python int, excluding bool")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
        if upper_bound is not None and value >= upper_bound:
            raise ValueError(f"{name} must be less than {upper_bound}")

    def _seed_sequence(
        self,
        stream: StreamId,
        *,
        variation: int = 0,
        index: int = 0,
    ) -> np.random.SeedSequence:
        """Derive fresh seed material from a validated identity."""
        self.validate(stream, variation=variation, index=index)

        # Coordinates occupy fixed 32-bit words. The arbitrary-sized root
        # comes last. Keep field order and the explicit zero index permanent.
        return np.random.SeedSequence(
            entropy=[int(stream), variation, index, self.root_seed]
        )

    def numpy(
        self,
        stream: StreamId,
        *,
        variation: int = 0,
        index: int = 0,
    ) -> np.random.Generator:
        """Return a fresh NumPy generator using explicitly chosen PCG64."""
        sequence = self._seed_sequence(
            stream,
            variation=variation,
            index=index,
        )
        return np.random.Generator(np.random.PCG64(sequence))

    def seed(
        self,
        stream: StreamId,
        *,
        variation: int = 0,
        index: int = 0,
    ) -> int:
        """Return a deterministic Python integer in [0, 2**64).

        Suitable for Torch manual_seed() and standard Gymnasium reset().
        Consumers may impose narrower limits or use fewer effective bits.
        """
        sequence = self._seed_sequence(
            stream,
            variation=variation,
            index=index,
        )
        return int(sequence.generate_state(1, dtype=np.uint64)[0])

    def torch(
        self,
        stream: StreamId,
        *,
        variation: int = 0,
        index: int = 0,
        device: torch.device | str = "cpu",
    ) -> torch.Generator:
        """Return a fresh Torch generator on the requested device.

        Use the same device as where the operation is going to be performed.
        Unsupported or unavailable devices fail through PyTorch;
        there is no automatic device fallback.
        """
        seed = self.seed(stream, variation=variation, index=index)
        return torch.Generator(device=device).manual_seed(seed)
