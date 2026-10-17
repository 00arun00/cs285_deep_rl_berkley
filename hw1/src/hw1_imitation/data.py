"""Load demonstrations and turn episode-local windows into training samples.

[`load_episodes_dataset`][hw1_imitation.data.load_episodes_dataset] reads the
packed Push-T arrays into [`Episode`][hw1_imitation.data.Episode] objects,
exposed through [`EpisodesDataset`][hw1_imitation.data.EpisodesDataset]. Use
[`download_pusht`][hw1_imitation.data.download_pusht] first if the dataset is
not already available locally.

Split whole episodes into training and validation subsets. For a fresh run,
fit [`Normalizer`][hw1_imitation.data.Normalizer] on the training subset; when
initializing from a checkpoint, reuse its stored normalizer. Use the same
statistics for validation and inference. Then build an
[`EpisodeChunkDataset`][hw1_imitation.data.EpisodeChunkDataset] for each subset:
it creates [`SingleEpisodeChunks`][hw1_imitation.data.SingleEpisodeChunks]
children and combines their samples without crossing episode boundaries.
Each sample pairs one state with a window of actions; optional padding
repeats the episode's final action to complete short windows.

Episode arrays can share backing storage. Construction marks the supplied
arrays read-only, but writable aliases can still change their contents.
Chunk datasets may borrow that storage, so keep source data unchanged while
using them. Every sampled state/action pair contains independent float32
tensors, even when the stored arrays are borrowed.
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeAlias

import numpy as np
import torch
import zarr
from numpy.typing import NDArray
from torch.utils.data import ConcatDataset, Dataset

PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZARR_RELATIVE_PATH = Path("pusht") / "pusht_cchi_v7_replay.zarr"

Float32Array: TypeAlias = NDArray[np.float32]


def download_pusht(dataset_dir: Path) -> Path:
    """Download and extract the Push-T dataset if needed.

    Returns the path to the extracted Zarr dataset.
    """

    dataset_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = dataset_dir / ZARR_RELATIVE_PATH
    # check if the final extracted path exists
    # exit early if final path exists
    # BUG: path might exist without final data
    # need some real data based checks.
    if zarr_path.exists():
        return zarr_path

    zip_path = dataset_dir / "pusht.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(PUSHT_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    return zarr_path


def _load_pusht_zarr(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[Helper] Load packed Push-T arrays from a Zarr dataset into memory.

    Args:
        zarr_path: Path to the extracted Push-T Zarr group.

    Returns:
        States and actions as float32 arrays, followed by exclusive episode
        end offsets as an int64 array. States and actions concatenate all
        episodes along their first axis.

    The Zarr group is opened read-only. The returned NumPy arrays are
    materialized in memory; they are not live views into the Zarr store.
    """
    root = zarr.open_group(zarr_path, mode="r")
    states_array = root["data/state"]
    actions_array = root["data/action"]
    episode_ends_array = root["meta/episode_ends"]

    assert isinstance(states_array, zarr.Array)
    assert isinstance(actions_array, zarr.Array)
    assert isinstance(episode_ends_array, zarr.Array)

    states = np.asarray(states_array, dtype=np.float32)  # 25650x5
    actions = np.asarray(actions_array, dtype=np.float32)  # 25650x2
    episode_ends = np.asarray(episode_ends_array, dtype=np.int64)  # 206x1
    return states, actions, episode_ends


@dataclass(frozen=True, slots=True)
class Episode:
    """Store the aligned state-action arrays for one demonstration.

    Each row pairs `states[t]` with `actions[t]`. Construct an episode from
    nonempty two-dimensional float32 arrays with matching timestep counts.
    Feature columns may be empty; finite values and `episode_id` are not
    validated here.

    Construction marks the supplied array objects read-only without copying
    their data. Frozen fields prevent rebinding, but writable aliases to the
    backing memory can still change values. This is a read-only interface,
    not deeply immutable storage. Keep the data unchanged while datasets use it.

    Attributes:
        episode_id: Caller-supplied identifier retained for dataset metadata.
            It need not equal the episode's position in a collection.
        states: State features with shape `(timesteps, state_dim)`, dtype
            float32. The supplied array object becomes read-only.
        actions: Actions with shape `(timesteps, action_dim)`, dtype float32.
            The supplied array object becomes read-only.

    Raises:
        ValueError: Either array is not two-dimensional, timestep counts differ,
            or there are no timesteps.
        TypeError: Either array has a dtype other than float32.
    """

    episode_id: int
    states: Float32Array  # shape: (timesteps, state_dim) - ReadOnly
    actions: Float32Array  # shape: (timesteps, action_dim) - ReadOnly

    def __post_init__(self) -> None:
        """Validate array shape and dtype, then mark both array objects read-only.

        This dataclass lifecycle hook runs on construction. Validation finishes
        before writeability flags are changed; no data is copied. Calling the hook
        again repeats these checks and flag updates.

        Raises:
            ValueError: Arrays are not two-dimensional, have different timestep
                counts, or contain no timesteps.
            TypeError: States or actions do not use float32.
        """
        if self.states.ndim != 2:
            raise ValueError(
                f"states should have shape: (timesteps, state_dim) but got {self.states.shape}"
            )

        if self.actions.ndim != 2:
            raise ValueError(
                f"actions should have shape: (timesteps, action_dim) but got {self.actions.shape}"
            )

        if len(self.states) != len(self.actions):
            raise ValueError("states and actions should be of same number of timesteps")

        if len(self.states) == 0:
            raise ValueError("episodes must have data for at least 1 timestamp")

        if self.actions.dtype != np.float32:
            raise TypeError(
                f"actions should have dtype:float32 but got {self.actions.dtype}"
            )

        if self.states.dtype != np.float32:
            raise TypeError(
                f"states should have dtype:float32 but got {self.states.dtype}"
            )

        # Make states and actions read_only.
        self.states.setflags(write=False)
        self.actions.setflags(write=False)

    def __len__(self) -> int:
        """Return the number of aligned timesteps in the episode.

        Returns:
            count: Number of rows in `states`, which matches `actions` after
                successful construction.
        """
        return len(self.states)


# Needed wrapper for utilizing pytorch random_split()
class EpisodesDataset(Dataset[Episode]):
    """Expose an ordered collection of episodes as a PyTorch dataset.

    Wrap episodes before using PyTorch's dataset utilities, such as
    `random_split`. The wrapper stores the supplied tuple by reference and
    returns the same episode objects. It performs no additional validation
    or copying; an empty collection is allowed.

    This implements the length and indexing interface of
    [PyTorch Dataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).
    It indexes whole demonstrations, not training windows; use
    [`EpisodeChunkDataset`][hw1_imitation.data.EpisodeChunkDataset] for windows.

    Attributes:
        episodes: The original ordered tuple of episode objects. Array ownership
            and writeability remain those of each episode.
    """
    def __init__(self, episodes: tuple[Episode, ...]):
        """Retain the episode collection without validating or copying it.

        Args:
            episodes: Ordered tuple to expose through the dataset. Empty tuples are
                valid; element types are assumed to match the annotation.
        """
        self.episodes = episodes

    def __len__(self) -> int:
        """Return the number of episodes in the stored collection.

        Returns:
            count: Length of `episodes`, including zero for an empty collection.
        """
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Episode:
        """Delegate episode lookup to the stored tuple.

        Args:
            idx: Integer position, with negative indices counted from the end.
                Although annotated for integer lookup, the implementation also
                passes slices through to the tuple unchanged.

        Returns:
            episode: The same stored episode object for an integer index. A slice
                returns a tuple of the original episode objects instead; no arrays
                are copied.

        Raises:
            IndexError: An integer index is outside the tuple's bounds.
            TypeError: The tuple rejects the index type.
        """
        return self.episodes[idx]


@dataclass(frozen=True, slots=True)
class Normalizer:
    """Normalize state and action features using stored means and scales.

    Use [`from_packed_data`][hw1_imitation.data.Normalizer.from_packed_data] or [`from_episodes_data`][hw1_imitation.data.Normalizer.from_episodes_data] to fit statistics on training
    observations, then reuse the same instance for validation and inference.
    [`normalize_state`][hw1_imitation.data.Normalizer.normalize_state] and [`normalize_action`][hw1_imitation.data.Normalizer.normalize_action] standardize features;
    [`denormalize_action`][hw1_imitation.data.Normalizer.denormalize_action] converts policy outputs back to action coordinates.

    Direct construction borrows the four supplied NumPy arrays without copying
    or changing their writeability. Frozen dataclass fields prevent rebinding,
    but array contents remain mutable, including through external aliases.
    Construction validates the statistics; transformations do not revalidate
    them. [`state_dict`][hw1_imitation.data.Normalizer.state_dict] and [`from_state_dict`][hw1_imitation.data.Normalizer.from_state_dict] copy storage for checkpoints.

    Attributes:
        state_mean: Finite per-feature state means, shape `(state_dim,)`.
        state_std: Finite, strictly positive state scales, matching `state_mean`.
        action_mean: Finite per-feature action means, shape `(action_dim,)`.
        action_std: Finite, strictly positive action scales, matching `action_mean`.

    Raises:
        ValueError: Statistics are empty, not matching one-dimensional vectors,
            contain nonfinite values, or have nonpositive scales.

    Examples:
        Fit only on training data and reuse the resulting statistics:

        ```python
        normalizer = Normalizer.from_episodes_data(training_episodes)
        normalized_states = normalizer.normalize_state(episode.states)
        normalized_actions = normalizer.normalize_action(episode.actions)
        actions = normalizer.denormalize_action(normalized_actions)
        ```
    """

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def __post_init__(self) -> None:
        """Validate statistics after dataclass construction.

        This lifecycle hook also runs when a factory constructs an instance.
        It neither copies nor freezes array storage.

        Raises:
            ValueError: A statistics vector is malformed or has invalid values.
        """
        self._validate_statistics()

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Standardize states by subtracting the mean and dividing by the scale.

        Args:
            state: State array, normally shaped `(..., state_dim)`. NumPy
                broadcasting determines compatible shapes; rank, feature count,
                dtype, and finite values are not checked explicitly.

        Returns:
            result: A newly computed array with the broadcast result shape. The input and
                stored statistics are not modified; NumPy determines the result dtype.

        Raises:
            ValueError: Input and statistics cannot be broadcast together.
        """
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Standardize actions by subtracting the mean and dividing by the scale.

        Args:
            action: Action array, normally shaped `(..., action_dim)`. NumPy
                broadcasting determines compatibility; rank, feature count, dtype,
                and finite values are not checked explicitly.

        Returns:
            result: A newly computed array with the broadcast result shape. The input and
                stored statistics are not modified; NumPy determines the result dtype.

        Raises:
            ValueError: Input and statistics cannot be broadcast together.
        """
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Restore action coordinates by multiplying by the scale and adding the mean.

        Args:
            action: Normalized action array, normally shaped `(..., action_dim)`.
                NumPy broadcasting determines compatibility; inputs are not
                explicitly checked for rank, feature count, dtype, or finite values.

        Returns:
            result: A newly computed array in the original action coordinates, with the
                broadcast result shape. Inputs and statistics are not modified.

        Raises:
            ValueError: Input and statistics cannot be broadcast together.
        """
        return action * self.action_std + self.action_mean

    @property
    def state_dim(self):
        """Return the number of state features in the stored mean vector.

        Returns:
            count (int): Number of entries in `state_mean`.
        """
        return self.state_mean.shape[0]

    @property
    def action_dim(self):
        """Return the number of action features in the stored mean vector.

        Returns:
            count (int): Number of entries in `action_mean`.
        """
        return self.action_mean.shape[0]

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Clamp scales from below without modifying the input (implementation detail).

        Args:
            std: Array of per-feature scales computed by a fitting method.
            eps: Lower bound applied elementwise. Positivity is assumed, not checked.

        Returns:
            result: A new array containing `maximum(std, eps)`. NaNs remain NaN and are
                rejected by subsequent construction validation.
        """
        return np.maximum(std, eps)

    @classmethod
    def from_packed_data(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> "Normalizer":
        """Fit feature statistics across the leading axis of packed training arrays.

        Means and population standard deviations (`ddof=0`) use NumPy's input-dtype
        rules. Scales are clamped to at least `1e-6`, including constant features.
        Input arrays are not modified; computed statistics have independent storage.

        Args:
            states: Training states, intended shape `(timesteps, state_dim)`.
            actions: Training actions, intended shape `(timesteps, action_dim)`.
                Equal timestep counts and two-dimensional inputs are assumptions,
                not explicit checks; each array is reduced independently.

        Returns:
            result: A normalizer containing the computed feature statistics.

        Raises:
            ValueError: Reduced statistics are not matching nonempty vectors or
                contain invalid values. Empty or nonfinite inputs can also emit
                NumPy warnings before validation fails.
        """
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    @classmethod
    def from_episodes_data(
        cls, episodes: Sequence[Episode] | Dataset[Episode]
    ) -> "Normalizer":
        """Fit statistics over all timesteps in a nonempty collection of episodes.

        Episodes are concatenated before fitting; longer episodes contribute more
        samples. This allocates combined state/action arrays, then delegates to
        `from_packed_data`. Episode arrays are not modified or retained as statistics.

        Args:
            episodes: Iterable sequence or dataset of episodes with compatible
                state feature dimensions and compatible action feature dimensions.

        Returns:
            result: A normalizer fitted to the concatenated demonstration data.

        Raises:
            ValueError: No episodes are supplied, feature shapes cannot be
                concatenated, or the resulting statistics fail validation.
        """
        states = [episode.states for episode in episodes]
        actions = [episode.actions for episode in episodes]

        if len(states) == 0 or len(actions) == 0:
            raise ValueError("Normalization requires at least one episode.")

        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        return cls.from_packed_data(states=states, actions=actions)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Export independent CPU tensors for the checkpoint schema.

        Revalidate statistics before export, since their arrays can have been
        mutated after construction. Each returned tensor owns copied numerical
        storage relative to this normalizer.

        Returns:
            result: Mapping with `state_mean`, `state_std`, `action_mean`, and `action_std`.
                Tensor shapes and supported NumPy dtypes are preserved.

        Raises:
            ValueError: Stored statistics no longer pass validation.
            TypeError: A statistics dtype is unsupported by `torch.from_numpy`.
        """
        self._validate_statistics()
        return {
            "state_mean": torch.from_numpy(self.state_mean.copy()),
            "state_std": torch.from_numpy(self.state_std.copy()),
            "action_mean": torch.from_numpy(self.action_mean.copy()),
            "action_std": torch.from_numpy(self.action_std.copy()),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "Normalizer":
        """Restore statistics without sharing checkpoint tensor storage.

        Each tensor is detached, moved to CPU, converted to NumPy, and copied.
        The four required keys belong to the checkpoint schema; extra keys are
        ignored. Construction validates the restored arrays. Schema changes must
        be coordinated with the outer checkpoint version.

        Args:
            state: Tensor mapping containing `state_mean`, `state_std`,
                `action_mean`, and `action_std`. Tensors must support NumPy conversion.

        Returns:
            result: A normalizer whose statistics are independent copies of the tensors.

        Raises:
            KeyError: A required key is absent.
            ValueError: Restored statistics fail construction validation.
            TypeError: A tensor dtype or layout cannot be converted to NumPy.
            RuntimeError: PyTorch rejects a tensor conversion, such as unresolved
                conjugate or negative flags.
        """

        def to_array(name: str) -> np.ndarray:
            return state[name].detach().cpu().numpy().copy()

        return cls(
            state_mean=to_array("state_mean"),
            state_std=to_array("state_std"),
            action_mean=to_array("action_mean"),
            action_std=to_array("action_std"),
        )

    def validate_dimensions(
        self,
        *,
        state_dim: int,
        action_dim: int,
    ) -> None:
        """Check statistics against the policy's expected feature dimensions.

        Only shapes are checked. This does not revalidate finite values or positive
        scales and does not modify the arrays.

        Args:
            state_dim: Expected number of state features.
            action_dim: Expected number of action features.

        Raises:
            ValueError: Any statistics array has a shape different from its
                expected one-dimensional feature shape.
        """
        expected_shapes = (
            ("state_mean", self.state_mean, (state_dim,)),
            ("state_std", self.state_std, (state_dim,)),
            ("action_mean", self.action_mean, (action_dim,)),
            ("action_std", self.action_std, (action_dim,)),
        )

        for name, values, expected in expected_shapes:
            if values.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {values.shape}"
                )

    def _validate_statistics(self) -> None:
        """Check vector shapes and finite values (implementation detail).

        Each mean must be a nonempty vector with a matching scale vector; scales
        must be strictly positive. No dtype requirement or copying is imposed.

        Raises:
            ValueError: A shape, finiteness, or positive-scale requirement fails.
        """
        for name, mean, std in (
            ("state", self.state_mean, self.state_std),
            ("action", self.action_mean, self.action_std),
        ):
            if mean.ndim != 1 or mean.size == 0 or std.shape != mean.shape:
                raise ValueError(f"{name} statistics must be matching nonempty vectors")
            if not np.isfinite(mean).all():
                raise ValueError(f"Invalid normalization statistics: {name}_mean")
            if not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError(f"Invalid normalization statistics: {name}_std")


def _get_episodes_from_packed_data(
    states: Float32Array,
    actions: Float32Array,
    episode_ends: NDArray[np.int64],
) -> tuple[Episode, ...]:
    """[Helper] Expose packed state-action arrays as read-only episode views.

    Args:
        states: Concatenated states with shape (total_timesteps, state_dim).
        actions: Concatenated actions with shape
            (total_timesteps, action_dim).
        episode_ends: Nonempty, one-dimensional integer array of exclusive
            end offsets. Offsets must be positive, strictly increasing,
            and end at total_timesteps.

    Returns:
        Episodes in source order, with IDs assigned from zero. Their arrays
        share memory with the inputs; numerical data is not copied.

    Raises:
        ValueError: Array shapes, lengths, or episode boundaries are invalid.
        TypeError: Episode boundaries are not integers, or states/actions
            do not use float32.
    """

    if len(states) != len(actions):
        raise ValueError("packed states and actions must have equal lengths")

    if episode_ends.ndim != 1:
        raise ValueError("episode_ends must be one-dimensional")

    if not np.issubdtype(episode_ends.dtype, np.integer):
        raise TypeError("episode_ends must contain integers")

    if len(episode_ends) == 0:
        raise ValueError("the dataset must contain at least one episode")

    if episode_ends[0] <= 0 or np.any(episode_ends[1:] <= episode_ends[:-1]):
        raise ValueError("episode ends must be positive and strictly increasing")

    if episode_ends[-1] != len(states):
        raise ValueError("the last episode must end at the end of the data")

    episode_start = 0
    episodes = []
    for episode_id, episode_end in enumerate(episode_ends):
        episode = Episode(
            episode_id=episode_id,
            states=states[episode_start:episode_end],
            actions=actions[episode_start:episode_end],
        )
        episodes.append(episode)
        episode_start = episode_end
    return tuple(episodes)


def load_episodes_dataset(zarr_path: Path) -> EpisodesDataset:
    """Load Push-T demonstrations as read-only episodes.

    Args:
        zarr_path: Path to the extracted Push-T Zarr group.

    Returns:
        EpisodesDataset: pytorch dataset containing Episodes

    The full state-action dataset is loaded into memory once. Episodes
    share those backing arrays through slices, avoiding per-episode data
    copies. The views keep their backing storage alive after this function
    returns.

    Ordinary writes through episode arrays are rejected. This is protection
    against accidental mutation, not a guarantee of deep immutability.
    """
    states, actions, episode_ends = _load_pusht_zarr(zarr_path=zarr_path)
    # Basic slices share storage; Episode makes these slices read-only.
    return EpisodesDataset(
        _get_episodes_from_packed_data(
            states=states,
            actions=actions,
            episode_ends=episode_ends,
        )
    )


class SingleEpisodeChunks(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Expose one episode as current-state and future-action training samples.

    At timestep `t`, a sample pairs `states[t]` with actions beginning at `t`.
    Without padding, only complete windows exist: the dataset length is
    `max(0, timesteps - chunk_length + 1)`. With padding, every original timestep
    is a valid start and windows past the end repeat the final action. Padding
    does not create extra start positions.

    The constructor optionally normalizes all rows once, before padding. It
    borrows the episode's arrays when no transformation is needed; normalization
    allocates new state/action arrays and padding allocates a new action array.
    Keep source episode arrays unchanged for this dataset's lifetime. Each
    lookup returns independent float32 tensors, so mutating a returned sample
    does not change later samples or source arrays.

    This is a [PyTorch Dataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).
    Use [`EpisodeChunkDataset`][hw1_imitation.data.EpisodeChunkDataset] to combine
    multiple episodes without windows crossing boundaries.

    Attributes:
        _chunk_length: Implementation detail: positive number of actions in
            each sample, retained from the constructor.
        _length: Implementation detail: number of valid sample starts, computed
            once from episode length, chunk length and padding mode.
        _states: Implementation detail: state rows used for lookup. Borrowed
            from the episode or independently computed by normalization.
        _actions: Implementation detail: action rows used for windows. Borrowed
            unless normalization or padding allocates a replacement array.
        _episode_id: Implementation detail: original episode identifier exposed
            by the `episode_id` property.
    """

    def __init__(
        self,
        episode: Episode,
        *,
        chunk_length: int,
        pad_action_chunk: bool = False,
        normalizer: Normalizer | None = None,
    ) -> None:
        """Prepare episode-local windows, optionally normalizing and padding once.

        The normalizer's dimensions are checked against the episode. Its statistics
        are not revalidated for finite or positive values here. Padding repeats the
        last action after normalization, if normalization was requested. Neither
        source array is modified and the normalizer itself is not retained.

        Args:
            episode: Demonstration supplying aligned state/action arrays and its ID.
            chunk_length: Positive Python integer specifying actions per window.
                Boolean values are explicitly rejected.
            pad_action_chunk: Whether to repeat the final action so every episode
                timestep has a complete window.
            normalizer: Statistics to apply to the episode's states and actions.
                `None` leaves them in their original coordinates.

        Raises:
            TypeError: `chunk_length` is not a Python integer or is a boolean.
            ValueError: `chunk_length` is not positive, or the normalizer's state
                or action dimensions differ from the episode's.
        """
        if isinstance(chunk_length, bool) or not isinstance(chunk_length, int):
            raise TypeError("chunk_length must be an integer")
        if chunk_length <= 0:
            raise ValueError("chunk_length must be positive")

        states, actions = episode.states, episode.actions

        # These transformations produce new arrays without modifying
        # the episode. Compute them once, rather than for every window.
        if normalizer is not None:
            # Validate dimensions.
            normalizer.validate_dimensions(
                state_dim=states.shape[1],
                action_dim=actions.shape[1],
            )
            (states, actions) = (
                normalizer.normalize_state(states),
                normalizer.normalize_action(actions),
            )

        if pad_action_chunk and chunk_length > 1:
            # Append enough actions for a chunk starting at the final
            # timestep. Padding does not create additional sample starts.
            actions = np.pad(
                array=actions,
                pad_width=((0, chunk_length - 1), (0, 0)),
                mode="edge",
            )

        self._chunk_length = chunk_length
        self._length = (
            len(episode)
            if pad_action_chunk
            else max(0, len(episode) - chunk_length + 1)
        )
        self._states = states
        self._actions = actions
        self._episode_id = episode.episode_id

    @property
    def state_dim(self) -> int:
        """Return the feature count of the prepared state rows.

        Returns:
            count: Number of columns in `_states`.
        """
        return self._states.shape[1]

    @property
    def action_dim(self) -> int:
        """Return the feature count of the prepared action rows.

        Returns:
            count: Number of columns in `_actions`.
        """
        return self._actions.shape[1]

    @property
    def episode_id(self) -> int:
        """Return the original episode's identifier.

        Returns:
            identifier: Identifier copied from the episode during construction;
                it is not inferred from an index or validated here.
        """
        return self._episode_id

    def __len__(self) -> int:
        """Return the number of valid window starts.

        Returns:
            count: Stored sample count. This can be zero when the episode is shorter
                than the chunk length and padding is disabled.
        """
        return self._length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Handle -ve index : wrap around.
        """Copy one current-state and action-window pair into float32 tensors.

        Args:
            index: Integer start position. Negative indices count from the end of
                this dataset's valid starts; slices are not supported.

        Returns:
            sample: A pair containing a state tensor shaped `(state_dim,)` and an
                action tensor shaped `(chunk_length, action_dim)`. Both are new
                float32 tensors independent of stored arrays and other samples.
                Coordinates are normalized only if a normalizer was supplied.

        Raises:
            IndexError: The index is outside `[-len(self), len(self))`; every index
                fails when this dataset is empty. NumPy also raises IndexError for
                unsupported scalar indices such as an in-range float.
            TypeError: An index such as a slice cannot be compared with integers,
                or an unsupported type reaches NumPy slicing.
        """
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError("index out of bound")

        chunk_end = index + self._chunk_length

        state = self._states[index]
        action_chunk = self._actions[index:chunk_end]

        return (
            torch.tensor(data=state, dtype=torch.float32),
            torch.tensor(data=action_chunk, dtype=torch.float32),
        )


class EpisodeChunkDataset(ConcatDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Concatenate episode-local training windows without crossing boundaries.

    Build one [`SingleEpisodeChunks`][hw1_imitation.data.SingleEpisodeChunks]
    child for each episode, in input order. All children must share state and
    action feature dimensions. Unpadded short episodes may contribute zero
    samples, but at least one valid sample must remain across the collection.

    [PyTorch ConcatDataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.ConcatDataset)
    supplies `__len__`, `__getitem__`, the child `datasets` collection and
    cumulative index routing. These inherited implementations are not repeated
    here. Valid negative indices count from the combined dataset's end;
    an index below `-len(self)` raises `ValueError`, while a nonnegative index
    at or above `len(self)` raises `IndexError` in the installed implementation.

    Construction borrows source arrays or computes replacements as described
    by `SingleEpisodeChunks`; it does not copy the input episodes wholesale.
    Keep their source storage unchanged while the dataset is in use. Returned
    state/action tensors are independent float32 copies.

    Attributes:
        _state_dim: Implementation detail: state feature count shared by all
            children and exposed through `state_dim`.
        _action_dim: Implementation detail: action feature count shared by all
            children and exposed through `action_dim`.

    Examples:
        Fit statistics on the training subset, then construct normalized windows
        from that same subset:

        ```python
        normalizer = Normalizer.from_episodes_data(training_episodes)
        windows = EpisodeChunkDataset(
            training_episodes, chunk_length=8, normalizer=normalizer,
            pad_action_chunk=True,
        )
        state, actions = windows[0]
        ```
    """

    def __init__(
        self,
        episodes: Sequence[Episode] | Dataset[Episode],
        *,
        chunk_length: int,
        normalizer: Normalizer | None = None,
        pad_action_chunk: bool = False,
    ) -> None:
        """Construct compatible child datasets and require a nonempty combined result.

        Each child applies normalization and padding independently. Dimension
        consistency is checked even for episodes that produce zero windows.
        The original sequence is iterated once; the parent retains the child
        datasets rather than the sequence itself.

        Args:
            episodes: Nonempty sequence or dataset of episodes in concatenation order.
            chunk_length: Positive Python integer specifying actions per window.
                Passed to every child; boolean values are rejected.
            normalizer: Shared statistics applied to every child, or `None` for
                original state/action coordinates. Feature dimensions must match.
            pad_action_chunk: Whether each child repeats its own final action to
                complete windows beginning near its end.

        Raises:
            ValueError: No episodes are supplied, episode dimensions disagree,
                a child rejects the chunk length or normalizer dimensions, or no
                valid windows remain after all children are constructed.
            TypeError: A child rejects a non-integer or boolean chunk length.
        """
        children = []
        expected_dimensions: tuple[int, int] | None = None

        for episode in episodes:
            dimensions = (
                episode.states.shape[1],
                episode.actions.shape[1],
            )

            if expected_dimensions is None:
                expected_dimensions = dimensions
            elif dimensions != expected_dimensions:
                raise ValueError(
                    f"episode {episode.episode_id} has state/action dimensions "
                    f"{dimensions}; expected {expected_dimensions}"
                )

            children.append(
                SingleEpisodeChunks(
                    episode,
                    chunk_length=chunk_length,
                    normalizer=normalizer,
                    pad_action_chunk=pad_action_chunk,
                )
            )

        if expected_dimensions is None:
            raise ValueError("at least one episode is required")

        self._state_dim, self._action_dim = expected_dimensions

        super().__init__(children)

        if len(self) == 0:
            raise ValueError("no valid chunks: reduce chunk_length or enable padding")

    @property
    def state_dim(self) -> int:
        """Return the state feature count shared by every child dataset.

        Returns:
            count: State dimension recorded during construction.
        """
        return self._state_dim

    @property
    def action_dim(self) -> int:
        """Return the action feature count shared by every child dataset.

        Returns:
            count: Action dimension recorded during construction.
        """
        return self._action_dim
