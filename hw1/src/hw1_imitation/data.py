"""Dataset utilities for Push-T."""

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
    """State-action pairs from one demonstration.

    States and actions are aligned by timestep: states[t] is the state
    associated with actions[t]. Both arrays must be nonempty, two-dimensional
    float32 arrays with the same number of timesteps.

    Construction marks the supplied array objects read-only without copying
    their data. Writable aliases to the backing memory can still change the
    values; this is a read-only interface, not deeply immutable storage.
    """

    episode_id: int
    states: Float32Array  # shape: (timesteps, state_dim) - ReadOnly
    actions: Float32Array  # shape: (timesteps, action_dim) - ReadOnly

    def __post_init__(self) -> None:
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
            raise ValueError("episodes must have data for atleast 1 timestamp")

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
        return len(self.states)


# Needed wrapper for utilizing pytorch random_split()
class EpisodesDataset(Dataset[Episode]):
    def __init__(self, episodes: tuple[Episode, ...]):
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Episode:
        return self.episodes[idx]


@dataclass(frozen=True, slots=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def __post_init__(self) -> None:
        self._validate_statistics()

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean

    @property
    def state_dim(self):
        return self.state_mean.shape[0]

    @property
    def action_dim(self):
        return self.action_mean.shape[0]

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.maximum(std, eps)

    @classmethod
    def from_packed_data(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> "Normalizer":
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    @classmethod
    def from_episodes_data(
        cls, episodes: Sequence[Episode] | Dataset[Episode]
    ) -> "Normalizer":
        states = [episode.states for episode in episodes]
        actions = [episode.actions for episode in episodes]

        if len(states) == 0 or len(actions) == 0:
            raise ValueError("Normalization requires atleast one episode.")

        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        return cls.from_packed_data(states=states, actions=actions)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return independent CPU tensors for the checkpoint's explicit schema."""
        self._validate_statistics()
        return {
            "state_mean": torch.from_numpy(self.state_mean.copy()),
            "state_std": torch.from_numpy(self.state_std.copy()),
            "action_mean": torch.from_numpy(self.action_mean.copy()),
            "action_std": torch.from_numpy(self.action_std.copy()),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "Normalizer":
        """Restore validated statistics without sharing checkpoint storage.

        These four keys are part of the checkpoint format. Changes to
        this schema must be coordinated with the outer checkpoint version.
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
    """Expose one episode as current-state / future-action samples.

    Each sample contains:
        state: Shape (state_dim,).
        actions: Shape (chunk_length, action_dim).

    Without padding, only complete chunks are available. With padding,
    every timestep is available, repeating the final action as needed.

    Source episode arrays must remain unchanged for this dataset's lifetime.
    Normalization and padding allocate new arrays when needed; otherwise,
    the dataset borrows the episode's arrays.

    Returned tensors are independent copies.
    """

    def __init__(
        self,
        episode: Episode,
        *,
        chunk_length: int,
        pad_action_chunk: bool = False,
        normalizer: Normalizer | None = None,
    ) -> None:
        """Set up one episode with normalization and padding."""
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
        """Number of features in each state."""
        return self._states.shape[1]

    @property
    def action_dim(self) -> int:
        """Number of components in each action."""
        return self._actions.shape[1]

    @property
    def episode_id(self) -> int:
        return self._episode_id

    def __len__(self) -> int:
        """Return the number of valid chunk starts."""
        return self._length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Handle -ve index : wrap around.
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
    """Combine episode-local windows into one indexed dataset.

    Each child handles one episode, so windows cannot cross episode
    boundaries. ConcatDataset supplies length and global index routing.

    Raises:
        ValueError: No episodes are supplied, dimensions differ between children,
            or no valid windows exist.
    """

    def __init__(
        self,
        episodes: Sequence[Episode] | Dataset[Episode],
        *,
        chunk_length: int,
        normalizer: Normalizer | None = None,
        pad_action_chunk: bool = False,
    ) -> None:
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
        """State dimension shared by all child datasets."""
        return self._state_dim

    @property
    def action_dim(self) -> int:
        """Action dimension shared by all child datasets."""
        return self._action_dim
