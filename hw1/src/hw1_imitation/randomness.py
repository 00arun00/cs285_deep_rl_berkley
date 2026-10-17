"""Assign stable identities to reproducible random streams.

**Identity and lifecycle**

1. Choose a permanent responsibility from [StreamId][hw1_imitation.randomness.StreamId].
2. Create a [RandomStreamFactory][hw1_imitation.randomness.RandomStreamFactory]
   with the experiment's root seed.
3. Select a variation and explicit peer index, then request a NumPy generator,
   Torch generator, or integer seed.

The identity is `(root_seed, stream_id, variation, index)`. Each generator
request starts a fresh stream; retain the returned object to advance it or
recreate the same identity to replay it. Global RNG state is not modified.

Registry values and seed-material encoding are permanent contracts. This
module controls stream assignment, not identical numerical results across
devices or library versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique

import numpy as np
import torch

_UINT32_LIMIT = 1 << 32  # 2**32


@unique
class StreamId(IntEnum):
    """Name the permanent integer IDs for distinct randomness responsibilities.

    **Registry rules**

    - Assign IDs explicitly; declaration order does not affect seed derivation.
    - Preserve an ID when renaming a member, and never reuse a retired ID.
    - Add a new ID for an independent responsibility.

    Attributes:
        DATA_SPLIT: Select training and validation episode membership.
        MODEL_INIT: Initialize model weights independently of training randomness.
        TRAIN_LOADER: Shuffle training samples and isolate training-loader randomness.
        VALIDATION_LOADER: Isolate validation-loader randomness.
        TRAIN_LOSS: Sample training-loss noise, times, and active dropout.
        VALIDATION_LOSS: Sample stochastic validation losses separately from training.
        EVAL_ENV: Seed environments used for periodic evaluation.
        EVAL_POLICY: Sample policy actions during periodic evaluation.
        FINAL_EVAL_ENV: Seed environments for final evaluation of the selected model.
        FINAL_EVAL_POLICY: Sample policy actions during that final evaluation.
    """

    DATA_SPLIT = 1  # Choose train/validation episode membership
    MODEL_INIT = 2  # Initialize model weights

    TRAIN_LOADER = 3  # Shuffle training samples and isolate loader randomness
    VALIDATION_LOADER = 4  # Isolate validation-loader randomness

    TRAIN_LOSS = 5
    VALIDATION_LOSS = 6

    EVAL_ENV = 7
    EVAL_POLICY = 8

    FINAL_EVAL_ENV = 9
    FINAL_EVAL_POLICY = 10

    # NEXT_ID = 11


@dataclass(frozen=True, slots=True)
class RandomStreamFactory:
    """Derive fresh generators and integer seeds from explicit stream coordinates.

    **Coordinates**

    - `stream` selects a permanent `StreamId` responsibility.
    - `variation` selects a realization of that responsibility; default zero.
    - `index` identifies a peer, such as an evaluation episode; default zero.
      It never increments automatically.

    **Reproducibility boundaries**

    - Every generator request returns a fresh object at its starting state. Keep
      that object for sequential draws; recreating it replays the stream.
    - No global RNG state is changed. The factory stores no advancing stream state.
    - Replay assumes unchanged encoding, library behavior, device, sampling
      operations, and arguments. Distinct identities can theoretically collide,
      especially when reduced to integer seeds.

    Attributes:
        root_seed: Nonnegative Python integer shared by derived identities. Booleans
            and non-Python integer types are rejected; there is no upper bound.
    """

    root_seed: int

    def __post_init__(self) -> None:
        """Validate the root seed when the frozen dataclass is constructed.

        No generator or seed material is created by this lifecycle hook.

        Raises:
            TypeError: `root_seed` is not a Python integer or is a boolean.
            ValueError: `root_seed` is negative.
        """
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

        Args:
            stream: A `StreamId` member, not its raw integer value. Its numeric ID must
                fit in an unsigned 32-bit word.
            variation: Python integer in `[0, 2**32)`, excluding booleans.
            index: Python integer in `[0, 2**32)`, excluding booleans.

        Raises:
            TypeError: The stream is not a `StreamId`, or a coordinate has an unsupported
                type, including a boolean or NumPy integer.
            ValueError: An integer coordinate is outside `[0, 2**32)`.
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
        """Check a Python integer without coercion (implementation detail).

        Args:
            value: Integer to check; booleans are explicitly excluded.
            name: Coordinate name included in validation error messages.
            upper_bound: Exclusive limit, or `None` for no upper limit. This helper
                assumes the supplied bound is suitable for comparison.

        Raises:
            TypeError: `value` is not a Python integer or is a boolean.
            ValueError: `value` is negative or reaches the supplied upper bound.
        """
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
        """Build fresh NumPy seed material from a validated identity.

        This implementation detail encodes entropy in the permanent order
        `[int(stream), variation, index, root_seed]`. Fixed 32-bit coordinates come
        first; the arbitrary-sized root comes last. Preserve the explicit zero index
        and field order to keep existing streams reproducible.

        Args:
            stream: Registered `StreamId` responsibility.
            variation: Realization coordinate in `[0, 2**32)`.
            index: Explicit peer coordinate in `[0, 2**32)`.

        Returns:
            sequence: New `SeedSequence` initialized from the identity, without changing
                the factory or any global random state.

        Raises:
            TypeError: A stream or coordinate has an unsupported type.
            ValueError: A coordinate is outside the unsigned 32-bit range.
        """
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
        """Return a fresh NumPy generator using the explicit PCG64 bit generator.

        Reuse the returned generator for successive draws. Another call with the same
        coordinates starts again rather than continuing the earlier object.

        Args:
            stream: Registered `StreamId` responsibility.
            variation: Python integer realization in `[0, 2**32)`, excluding booleans.
            index: Python integer peer index in `[0, 2**32)`, excluding booleans.

        Returns:
            generator: Independent generator initialized from the derived seed sequence.
                Its draws do not advance another generator or the global NumPy RNG.

        Raises:
            TypeError: A stream or coordinate has an unsupported type.
            ValueError: A coordinate is outside the unsigned 32-bit range.
        """
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
        """Return a deterministic unsigned 64-bit seed as a Python integer.

        Generate one `uint64` word from a fresh seed sequence. Suitable consumers include
        Torch `manual_seed()` and standard Gymnasium `reset()`; consumers may impose
        narrower limits or use fewer effective bits. Distinct identities are not
        mathematically guaranteed to produce distinct reduced seeds.

        Args:
            stream: Registered `StreamId` responsibility.
            variation: Python integer realization in `[0, 2**32)`, excluding booleans.
            index: Python integer peer index in `[0, 2**32)`, excluding booleans.

        Returns:
            seed: Python integer in `[0, 2**64)`, repeated for the same identity under
                unchanged seed derivation behavior.

        Raises:
            TypeError: A stream or coordinate has an unsupported type.
            ValueError: A coordinate is outside the unsigned 32-bit range.
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

        Seed it with [seed][hw1_imitation.randomness.RandomStreamFactory.seed] for the
        same coordinates. The device does not change the derived integer seed, but
        sampling results can differ by device. Match the generator to the operation's
        sampling device; unsupported or unavailable devices fail through PyTorch with
        no automatic fallback.

        Args:
            stream: Registered `StreamId` responsibility.
            variation: Python integer realization in `[0, 2**32)`, excluding booleans.
            index: Python integer peer index in `[0, 2**32)`, excluding booleans.
            device: Device accepted by `torch.Generator`, defaulting to CPU.

        Returns:
            generator: New, manually seeded Torch generator. Retain it to advance the
                stream; another request with the same coordinates starts a replay.

        Raises:
            TypeError: A stream or coordinate has an unsupported type.
            ValueError: A coordinate is outside the unsigned 32-bit range.
        """
        seed = self.seed(stream, variation=variation, index=index)
        return torch.Generator(device=device).manual_seed(seed)
