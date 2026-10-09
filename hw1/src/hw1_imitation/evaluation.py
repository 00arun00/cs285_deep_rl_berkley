"""Evaluation utilities for Push-T policies."""

from __future__ import annotations

import logging
import math
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from hw1_imitation.checkpoint import CHECKPOINT_VERSION, save_policy
from hw1_imitation.data import Normalizer
from hw1_imitation.logging_utils import open_video_writer
from hw1_imitation.model import BasePolicy
from hw1_imitation.randomness import RandomStreamFactory, StreamId

NUM_EVAL_EPISODES = 100


@dataclass(frozen=True, slots=True)
class EvaluationResults:
    """Measurements and completed video paths from one evaluation.

    Paths refer to persistent files owned by the experiment.
    Consumers may read them but do not acquire deletion ownership.
    """

    mean_reward: float
    num_episodes: int
    video_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ValidationResults:
    loss_mean: float
    examples_count: int


def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(frame)
    resized = image.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(resized)


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    score: float
    path: Path
    artifact: wandb.Artifact | None = None


def prune_checkpoints(
    checkpoints: dict[int, CheckpointRecord],
    keep_steps: set[int],
) -> set[int]:
    """Delete obsolete checkpoints; return steps whose cleanup failed."""
    pending: set[int] = set()
    for step, record in list(checkpoints.items()):
        if step in keep_steps:
            continue

        try:
            record.path.unlink(missing_ok=True)
        except OSError:
            logging.exception("Could not delete local checkpoint at step %s", step)
            pending.add(step)
            continue

        try:
            # Preserve W&B's protection for user-assigned aliases.
            if record.artifact is not None:
                record.artifact.delete()
        except Exception:
            logging.exception("Could not delete W&B checkpoint at step %s", step)
            pending.add(step)
            continue

        # Failed deletions stay tracked for the next pruning pass.
        del checkpoints[step]

    return pending


def save_checkpoint_and_retain(
    model: BasePolicy,
    step: int,
    *,
    normalizer: Normalizer,
    flow_num_steps: int,
    mean_reward: float,
    top_k: int,
    checkpoints: dict[int, CheckpointRecord],
    checkpoint_dir: Path,
    run: wandb.Run | None = None,
) -> set[int]:
    """Save an evaluated policy and retain the best K plus latest.

    Call once per evaluation with increasing steps and a fixed top_k.
    Higher rewards win; ties favor earlier steps. Zero keeps only latest.
    A supplied W&B run must be online; omit it for local-only saving.

    When uploading, completion is confirmed before pruning. Cleanup failures are
    logged and retried at later saves; save/upload errors propagate.
    Returns pending cleanup steps. Retry state is not persisted across runs.
    """
    if run is not None and run.offline:
        raise ValueError("Remote checkpoint retention requires online W&B.")
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    if not math.isfinite(mean_reward):
        raise ValueError("Checkpoint mean_reward must be finite")

    checkpoint_path = checkpoint_dir / f"policy_step_{step}.pt"
    save_policy(
        path=checkpoint_path,
        model=model,
        normalizer=normalizer,
        flow_num_steps=flow_num_steps,
    )

    # Use the same retention decision for local files and remote versions.
    scores = {saved_step: record.score for saved_step, record in checkpoints.items()}
    scores[step] = mean_reward
    ranked_steps = sorted(
        scores, key=lambda saved_step: (-scores[saved_step], saved_step)
    )
    keep_steps = set(ranked_steps[:top_k]) | {step}

    uploaded = None
    if run is not None:
        artifact = wandb.Artifact(
            name=f"policy-checkpoint-{run.id}",
            type="model",
            metadata={
                "step": step,
                "mean_reward": mean_reward,
                "format_version": CHECKPOINT_VERSION,
            },
        )
        artifact.add_file(checkpoint_path.as_posix(), name=checkpoint_path.name)

        # Logging moves these aliases; it does not remove older versions.
        aliases = ["latest"]
        if top_k > 0 and ranked_steps[0] == step:
            aliases.append("best")
        uploaded = run.log_artifact(artifact, aliases=aliases)
        uploaded.wait()
    checkpoints[step] = CheckpointRecord(mean_reward, checkpoint_path, uploaded)

    return prune_checkpoints(checkpoints, keep_steps)


@torch.no_grad()
def compute_validation_loss(
    model: BasePolicy,
    loader: DataLoader,
    *,
    device: torch.device,
    generator: torch.Generator,
    show_progress: bool = False,
    progress_position: int = 0,
) -> ValidationResults:
    """Compute example-weighted loss using caller-owned randomness.

    The generator advances across batches and must match the sampling device.
    To repeat validation noise across checkpoints, the caller should provide
    a fresh generator with the same identity for each complete validation pass.

    Repeating noise also assumes unchanged batch order and sampling shapes.
    The model's previous training mode is restored on exit.
    """

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_examples = 0

    try:
        with tqdm(
            total=len(loader),
            desc="Validate",
            unit="batch",
            position=progress_position,
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
            miniters=1,
            disable=None if show_progress else True,
            postfix={"loss": "—"},
        ) as progress:
            for states, action_chunks in loader:
                states = states.to(device)
                action_chunks = action_chunks.to(device)

                loss = model.compute_loss(
                    state=states,
                    action_chunk=action_chunks,
                    generator=generator,
                )

                batch_size = states.shape[0]
                total_loss += loss.item() * batch_size
                total_examples += batch_size

                progress.set_postfix(
                    {"loss": f"{total_loss / total_examples:.4g}"},
                    refresh=False,
                )
                progress.update(1)

        if total_examples == 0:
            raise ValueError("validation loader produced no examples")

        return ValidationResults(
            loss_mean=total_loss / total_examples,
            examples_count=total_examples,
        )
    finally:
        model.train(was_training)


def evaluate_policy(
    model: BasePolicy,
    normalizer: Normalizer,
    device: torch.device,
    chunk_size: int,
    video_size: tuple[int, int],
    num_video_episodes: int,
    flow_num_steps: int,
    *,
    streams: RandomStreamFactory,
    env_variation: int = 0,
    policy_variation: int = 0,
    video_dir: Path | None = None,
    num_eval_episodes: int = NUM_EVAL_EPISODES,
    show_progress: bool = False,
    progress_position: int = 0,
) -> EvaluationResults:
    """Run policy rollouts and return scores for Push-T state observations.

    The score is the mean of per-episode maximum rewards.

    Each episode index identifies an environment seed and a separate policy
    generator. Repeated calls with the same root and variations recreate
    those episode identities.

    The policy generator advances across action chunks within its episode.
    Environment and policy variations can be changed independently.

    Videos are streamed directly to the caller-provided directory.
    A failed recording may leave a partial file, but its path is never
    returned as a successful result.

    This function owns and closes the environment. It restores the
    model's previous top-level training mode on exit.

    Args:
        model: Policy producing normalized action chunks.
        normalizer: Observation and action normalization statistics.
        device: Device used for policy inference.
        chunk_size: Actions executed before requesting another chunk.
        video_size: Recording dimensions as (width, height).
        num_video_episodes: Initial episodes to record, capped by the
            evaluation episode count. Use zero to disable recording.
        flow_num_steps: Sampling steps for flow policies.
        streams: Stateless factory supplying indexed episode randomness.
        env_variation: Realization of evaluation environment randomness.
        policy_variation: Realization of evaluation policy randomness.
        video_dir: Existing directory required when recording videos.
        num_eval_episodes: Positive number of rollout episodes.
        show_progress: Show a temporary bar on an interactive terminal.
        progress_position: Terminal row assigned to the progress bar.

    Invalid configuration raises ValueError or NotADirectoryError.
    Environment, filesystem, and encoding errors propagate.
    """
    if num_eval_episodes <= 0:
        raise ValueError("num_eval_episodes must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if num_video_episodes < 0:
        raise ValueError("num_video_episodes must be non-negative")

    if num_video_episodes > 0:
        if video_dir is None:
            raise ValueError("video_dir is required when recording videos")
        if not video_dir.is_dir():
            raise NotADirectoryError(video_dir)
        width, height = video_size
        if width <= 0 or height <= 0:
            raise ValueError("video_size dimensions should be positive")
        if width % 2 != 0 or height % 2 != 0:
            raise ValueError("video_size dimensions must be even for yuv420p encoding")

    rewards: list[float] = []
    video_paths: list[Path] = []
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="state",
        render_mode="rgb_array",
    )

    # Register cleanup before model inference or env stepping.
    # There are evealuated LIFO.
    with ExitStack() as resources:
        resources.callback(env.close)
        resources.callback(model.train, model.training)
        model.eval()

        action_space = env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("Evaluation requires a Box action space")

        action_low = action_space.low
        action_high = action_space.high

        progress = resources.enter_context(
            tqdm(
                total=num_eval_episodes,
                desc="Evaluate",
                unit="episode",
                position=progress_position,
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
                miniters=1,
                disable=None if show_progress else True,
                postfix={"score": "—"},
            )
        )

        for episode_index in range(num_eval_episodes):
            # Give each episode separately indexed environment and policy streams.
            env_seed = streams.seed(
                StreamId.EVAL_ENV,
                variation=env_variation,
                index=episode_index,
            )

            policy_generator = streams.torch(
                StreamId.EVAL_POLICY,
                variation=policy_variation,
                index=episode_index,
                device=device,
            )

            obs, _ = env.reset(seed=env_seed)
            done = False
            chunk_index = chunk_size
            action_chunk: np.ndarray | None = None
            max_reward = -np.inf

            video_path = None
            if video_dir is not None and episode_index < num_video_episodes:
                video_path = video_dir / f"episode_{episode_index:07d}.mp4"

            recording_context = (
                open_video_writer(video_path, fps=20)
                if video_path is not None
                else nullcontext(None)
            )

            with recording_context as writer:
                while not done:
                    if action_chunk is None or chunk_index >= chunk_size:
                        state = (
                            torch.from_numpy(normalizer.normalize_state(obs))
                            .float()
                            .to(device)
                        )
                        with torch.no_grad():
                            predicted = (
                                model.sample_actions(
                                    state.unsqueeze(0),
                                    generator=policy_generator,
                                    num_steps=flow_num_steps,
                                )
                                .cpu()
                                .numpy()[0]
                            )

                        action_chunk = normalizer.denormalize_action(predicted)
                        expected_shape = action_space.shape
                        if (
                            action_chunk.ndim != len(expected_shape) + 1
                            or action_chunk.shape[1:] != expected_shape
                        ):
                            raise ValueError(
                                f"Expected action chunk shaped (horizon, {expected_shape}),"
                                f"got {action_chunk.shape}"
                            )

                        if action_chunk.shape[0] < chunk_size:
                            raise ValueError(
                                f"chunk_size={chunk_size} exceeds the predicted "
                                f"action horizon={action_chunk.shape[0]}"
                            )
                        if not np.isfinite(action_chunk).all():
                            raise ValueError(
                                "Policy produced non-finite actions after denormalization"
                            )

                        action_chunk = np.clip(
                            action_chunk,
                            action_low,
                            action_high,
                        )
                        chunk_index = 0

                    action = action_chunk[chunk_index]
                    obs, reward, terminated, truncated, info = env.step(
                        action.astype(np.float32)
                    )
                    if writer is not None:
                        rendered = env.render()
                        if not isinstance(rendered, np.ndarray):
                            raise TypeError("Expected RGB array from env.render()")

                        frame = resize_frame(
                            frame=rendered,
                            size=video_size,
                        )
                        writer.append_data(frame)

                    max_reward = max(max_reward, float(reward))
                    done = terminated or truncated
                    chunk_index += 1

            rewards.append(max_reward)

            # Reached only after the rollout and writer clouser succeeded.
            if video_path is not None:
                video_paths.append(video_path)

            progress.set_postfix(
                {"score": f"{float(np.mean(rewards)):.3f}"},
                refresh=False,
            )
            progress.update(1)

    return EvaluationResults(
        mean_reward=float(np.mean(rewards)),
        num_episodes=len(rewards),
        video_paths=tuple(video_paths),
    )
