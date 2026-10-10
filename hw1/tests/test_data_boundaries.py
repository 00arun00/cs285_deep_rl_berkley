"""Episode-local windows preserve boundaries, indexing, and source storage."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
import zarr
from hw1_imitation.data import (
    Episode,
    EpisodeChunkDataset,
    Normalizer,
    SingleEpisodeChunks,
    load_episodes_dataset,
)
from hypothesis import example, given, settings
from hypothesis import strategies as st


def tagged_episode(episode_id, length, state_dim, action_dim):
    """Distinct episode/timestep/feature values expose misrouted windows."""
    states = (
        (episode_id * 1000 + np.arange(length * state_dim))
        .reshape(length, state_dim)
        .astype(np.float32)
    )
    actions = (
        (episode_id * 1000 + 100 + np.arange(length * action_dim))
        .reshape(length, action_dim)
        .astype(np.float32)
    )
    return Episode(episode_id, states, actions)


def reference_samples(episodes, horizon, pad):
    """Enumerate windows directly from source episodes, without dataset helpers."""
    samples = []
    for episode in episodes:
        for start in range(len(episode)):
            if not pad and start + horizon > len(episode):
                continue
            actions = np.stack(
                [
                    episode.actions[min(start + offset, len(episode) - 1)]
                    for offset in range(horizon)
                ]
            )
            samples.append((episode.states[start], actions))
    return samples


def assert_samples(dataset, expected):
    assert len(dataset) == len(expected)
    for index, (expected_state, expected_actions) in enumerate(expected):
        # Every legal negative index aliases its corresponding positive index.
        for access_index in (index, index - len(expected)):
            state, actions = dataset[access_index]
            assert state.dtype == actions.dtype == torch.float32
            np.testing.assert_array_equal(state.numpy(), expected_state)
            np.testing.assert_array_equal(actions.numpy(), expected_actions)


@pytest.mark.parametrize(
    "pad", [False, True], ids=["complete-windows", "padded-windows"]
)
@given(
    length=st.integers(1, 8),
    horizon=st.integers(1, 10),
    state_dim=st.integers(1, 4),
    action_dim=st.integers(1, 4),
)
@example(length=1, horizon=1, state_dim=1, action_dim=1)
@example(length=1, horizon=4, state_dim=3, action_dim=2)
@example(length=4, horizon=4, state_dim=3, action_dim=2)
def test_single_episode_windows_and_index_bounds(
    pad, length, horizon, state_dim, action_dim
):
    episode = tagged_episode(0, length, state_dim, action_dim)
    dataset = SingleEpisodeChunks(episode, chunk_length=horizon, pad_action_chunk=pad)
    assert_samples(dataset, reference_samples([episode], horizon, pad))
    for index in (len(dataset), len(dataset) + 3, -len(dataset) - 1, -len(dataset) - 4):
        with pytest.raises(IndexError):
            dataset[index]


@pytest.mark.parametrize(
    "pad", [False, True], ids=["complete-windows", "padded-windows"]
)
@given(
    lengths=st.lists(st.integers(1, 8), min_size=1, max_size=5),
    horizon=st.integers(1, 10),
    state_dim=st.integers(1, 4),
    action_dim=st.integers(1, 4),
)
@example(lengths=[1, 4, 1, 3], horizon=3, state_dim=3, action_dim=2)
@example(lengths=[1, 2], horizon=4, state_dim=1, action_dim=1)
@example(lengths=[2, 2], horizon=1, state_dim=2, action_dim=3)
def test_combined_windows_never_cross_episode_boundaries(
    pad, lengths, horizon, state_dim, action_dim
):
    episodes = [
        tagged_episode(i, length, state_dim, action_dim)
        for i, length in enumerate(lengths)
    ]
    expected = reference_samples(episodes, horizon, pad)
    if not expected:
        with pytest.raises(ValueError, match="no valid chunks"):
            EpisodeChunkDataset(episodes, chunk_length=horizon, pad_action_chunk=pad)
        return

    dataset = EpisodeChunkDataset(episodes, chunk_length=horizon, pad_action_chunk=pad)
    assert_samples(dataset, expected)
    for index in (len(dataset), len(dataset) + 3):
        with pytest.raises(IndexError):
            dataset[index]
    # ConcatDataset supplies the negative-index contract for this wrapper.
    with pytest.raises(ValueError):
        dataset[-len(dataset) - 1]


@pytest.mark.parametrize("combined", [False, True], ids=["single", "combined"])
@pytest.mark.parametrize(
    "pad", [False, True], ids=["complete-windows", "padded-windows"]
)
@pytest.mark.parametrize("normalize", [False, True], ids=["raw", "normalized"])
@given(
    data=st.data(),
    length=st.integers(1, 8),
    state_dim=st.integers(1, 4),
    action_dim=st.integers(1, 4),
)
def test_returned_tensors_and_dataset_construction_preserve_source_storage(
    combined, pad, normalize, data, length, state_dim, action_dim
):
    horizon = data.draw(st.integers(1, 10 if pad else length), label="horizon")
    episodes = [
        tagged_episode(i, length, state_dim, action_dim)
        for i in range(2 if combined else 1)
    ]
    originals = [
        (episode.states.copy(), episode.actions.copy()) for episode in episodes
    ]
    normalizer = (
        Normalizer(
            np.full(state_dim, 3, dtype=np.float32),
            np.full(state_dim, 2, dtype=np.float32),
            np.full(action_dim, -5, dtype=np.float32),
            np.full(action_dim, 4, dtype=np.float32),
        )
        if normalize
        else None
    )
    options = dict(chunk_length=horizon, pad_action_chunk=pad, normalizer=normalizer)
    dataset = (
        EpisodeChunkDataset(episodes, **options)
        if combined
        else SingleEpisodeChunks(episodes[0], **options)
    )
    expected = reference_samples(episodes, horizon, pad)
    if normalize:
        expected = [((state - 3) / 2, (actions + 5) / 4) for state, actions in expected]
    assert_samples(dataset, expected)

    index = data.draw(st.integers(0, len(dataset) - 1), label="mutated_index")
    state, actions = dataset[index]
    state.fill_(float("nan"))
    actions.fill_(float("nan"))
    for episode, (original_states, original_actions) in zip(episodes, originals):
        np.testing.assert_array_equal(episode.states, original_states)
        np.testing.assert_array_equal(episode.actions, original_actions)
    # Check every window, including overlapping neighbors of the mutated one.
    assert_samples(dataset, expected)


def load_packed_dataset(states, actions, episode_ends):
    """Exercise the public loader against the on-disk Push-T schema.

    Allocate a fresh store per Hypothesis example, not per pytest invocation.
    No private loading/splitting helpers are called or mocked.
    """
    with TemporaryDirectory() as directory:
        path = Path(directory) / "demonstrations.zarr"
        group = zarr.open_group(path, mode="w")
        data = group.create_group("data")
        data.create_array("state", data=states)
        data.create_array("action", data=actions)
        group.create_group("meta").create_array("episode_ends", data=episode_ends)
        return load_episodes_dataset(path)


# Keep real filesystem examples small; I/O latency is not a tested property.
@settings(max_examples=40, deadline=None)
@given(
    lengths=st.lists(st.integers(1, 8), min_size=1, max_size=5),
    state_dim=st.integers(1, 4),
    action_dim=st.integers(1, 4),
)
@example(lengths=[1], state_dim=1, action_dim=1)
@example(lengths=[1, 4, 1, 3], state_dim=3, action_dim=2)
def test_packed_dataset_reconstructs_aligned_episodes(lengths, state_dim, action_dim):
    originals = [
        tagged_episode(i, length, state_dim, action_dim)
        for i, length in enumerate(lengths)
    ]
    states = np.concatenate([episode.states for episode in originals])
    actions = np.concatenate([episode.actions for episode in originals])
    ends = np.cumsum(lengths, dtype=np.int64)
    loaded = load_packed_dataset(states, actions, ends)

    assert len(loaded) == len(originals)
    for actual, expected in zip(loaded, originals):
        assert actual.episode_id == expected.episode_id
        np.testing.assert_array_equal(actual.states, expected.states)
        np.testing.assert_array_equal(actual.actions, expected.actions)
    # Reconstruction must account for every packed timestep, in source order.
    np.testing.assert_array_equal(
        np.concatenate([episode.states for episode in loaded]), states
    )
    np.testing.assert_array_equal(
        np.concatenate([episode.actions for episode in loaded]), actions
    )


@pytest.mark.parametrize(
    "corruption",
    ["duplicate", "decreasing", "zero", "final-before-end", "final-after-end"],
)
@settings(max_examples=40, deadline=None)
@given(lengths=st.lists(st.integers(2, 8), min_size=3, max_size=5))
def test_packed_dataset_rejects_invalid_episode_boundaries(corruption, lengths):
    total = sum(lengths)
    states = np.arange(total * 3, dtype=np.float32).reshape(total, 3)
    actions = np.arange(total * 2, dtype=np.float32).reshape(total, 2)
    ends = np.cumsum(lengths, dtype=np.int64)
    if corruption == "duplicate":
        ends[1] = ends[0]
    elif corruption == "decreasing":
        ends[0], ends[1] = ends[1], ends[0]
    elif corruption == "zero":
        ends[0] = 0
    elif corruption == "final-before-end":
        # Length >= 2 ensures this remains increasing: only the final offset is wrong.
        ends[-1] -= 1
    else:
        ends[-1] += 1
    with pytest.raises(ValueError):
        load_packed_dataset(states, actions, ends)


def test_packed_dataset_rejects_empty_input():
    with pytest.raises(ValueError):
        load_packed_dataset(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.array([], dtype=np.int64),
        )


def test_episode_rejects_empty_timesteps():
    with pytest.raises(ValueError):
        Episode(
            0, np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        )


def test_chunk_dataset_rejects_empty_episode_collection():
    with pytest.raises(ValueError):
        EpisodeChunkDataset([], chunk_length=1)


@pytest.mark.parametrize(
    "state_dim,action_dim",
    [(4, 2), (3, 4)],
    ids=["different-state-dimensions", "different-action-dimensions"],
)
def test_chunk_dataset_rejects_inconsistent_feature_dimensions(state_dim, action_dim):
    episodes = [tagged_episode(0, 3, 3, 2), tagged_episode(1, 3, state_dim, action_dim)]
    with pytest.raises(ValueError):
        EpisodeChunkDataset(episodes, chunk_length=1)
