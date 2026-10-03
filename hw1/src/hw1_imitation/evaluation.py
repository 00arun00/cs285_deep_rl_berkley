"""Evaluation utilities for Push-T policies."""

from __future__ import annotations

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


def log_checkpoint_artifact(
    model: BasePolicy,
    step: int,
    *,
    normalizer: Normalizer,
    flow_num_steps: int,
) -> None:
    """Save a reusable policy and upload it to the active W&B run."""
    if wandb.run is None:
        raise RuntimeError("wandb.init did not create a run.")

    checkpoint_dir = Path(wandb.run.dir) / "checkpoints"
    checkpoint_path = checkpoint_dir / f"policy_step_{step}.pt"

    save_policy(
        path=checkpoint_path,
        model=model,
        normalizer=normalizer,
        flow_num_steps=flow_num_steps,
    )

    artifact = wandb.Artifact(
        name=f"policy-checkpoint-{wandb.run.id}",
        type="model",
        metadata={
            "step": step,
            "format_version": CHECKPOINT_VERSION,
        },
    )
    artifact.add_file(
        checkpoint_path.as_posix(),
        name=checkpoint_path.name,
    )
    wandb.log_artifact(artifact)


@torch.no_grad()
def compute_validation_loss(
    model: BasePolicy,
    loader: DataLoader,
    *,
    device: torch.device,
    show_progress: bool = False,
    progress_position: int = 0,
) -> ValidationResults:
    """Compute example-weighted loss with an optional temporary batch bar."""
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
    video_dir: Path | None = None,
    num_eval_episodes: int = NUM_EVAL_EPISODES,
    show_progress: bool = False,
    progress_position: int = 0,
) -> EvaluationResults:
    """Run policy rollouts and return scores for Push-T state observations.

    The score is the mean of per-episode maximum rewards. Episodes use
    reset seeds 0 through num_eval_episodes - 1.

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
            obs, _ = env.reset(seed=episode_index)
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
                                    state.unsqueeze(0), num_steps=flow_num_steps
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
