"""Measure policy performance and retain evaluated checkpoints.

**Two different measurements**

- [compute_validation_loss][hw1_imitation.evaluation.compute_validation_loss]
  averages batch losses weighted by example count.
- [evaluate_policy][hw1_imitation.evaluation.evaluate_policy] runs Push-T
  rollouts and averages each episode's maximum reward, not its reward sum.

Both temporarily use evaluation mode and restore the model's prior top-level
training flag. Randomness is explicit: validation receives a generator, while
rollouts derive separate indexed environment and policy streams.

[save_checkpoint_and_retain][hw1_imitation.evaluation.save_checkpoint_and_retain]
keeps the best K evaluated checkpoints plus the latest. Video paths and retained
checkpoint records refer to experiment-owned files; callers manage their
longer-term lifecycle.
"""

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
    """Record rollout scores and successfully completed video paths.

    This frozen dataclass stores supplied values without validation. Consumers may
    read the persistent video files but do not acquire deletion ownership.

    Attributes:
        mean_reward: Arithmetic mean of per-episode maximum rewards, as produced by
            `evaluate_policy`. This is not mean cumulative episode return.
        num_episodes: Number of completed rollout episodes included in the score.
        video_paths: Paths appended only after their rollout and writer closure
            succeeded. The tuple may be empty when recording is disabled.
    """

    mean_reward: float
    num_episodes: int
    video_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ValidationResults:
    """Record an example-weighted validation loss and its sample count.

    This frozen dataclass stores values without additional validation.

    Attributes:
        loss_mean: Sum of batch mean losses multiplied by batch sizes, divided by
            the total number of examples.
        examples_count: Total examples processed, including a smaller final batch.
    """
    loss_mean: float
    examples_count: int


def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an image array using PIL bilinear interpolation.

    Args:
        frame: Array accepted by `PIL.Image.fromarray`, normally an RGB render frame.
            This helper does not validate a particular color layout or dtype.
        size: Destination `(width, height)` passed directly to PIL.

    Returns:
        resized: NumPy representation of the resized image. Shape and dtype follow
            the PIL image mode; callers should not assume the returned array is writable.

    PIL conversion and size errors propagate. The input array is not modified.
    """
    image = Image.fromarray(frame)
    resized = image.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(resized)


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Track a saved checkpoint's score, local path, and optional remote artifact.

    The record is frozen, but referenced files and remote artifacts have independent
    lifetimes. Construction does not validate the score or check file existence.

    Attributes:
        score: Evaluation score used for retention; higher scores rank first.
        path: Local checkpoint file managed by the retention helpers.
        artifact: Uploaded W&B artifact handle, or `None` for local-only saving.
            Cleanup uses the handle's normal alias-protection behavior.
    """
    score: float
    path: Path
    artifact: wandb.Artifact | None = None


def prune_checkpoints(
    checkpoints: dict[int, CheckpointRecord],
    keep_steps: set[int],
) -> set[int]:
    """Remove obsolete local and remote checkpoints, retaining failed cleanup records.

    **Cleanup order**

    1. Skip steps in `keep_steps` and unlink each other local path. Missing files
       count as success.
    2. Delete its W&B artifact, if present, without bypassing user-alias protection.
    3. Remove the dictionary entry only after both operations succeed.

    A local `OSError` or remote deletion exception is logged and leaves the entry
    for retry. A remote failure can therefore leave a record whose local file is
    already gone. The helper mutates the supplied dictionary, not a copy.

    Args:
        checkpoints: Mutable mapping of steps to experiment-owned checkpoint records.
        keep_steps: Steps exempt from deletion; absent steps have no effect.

    Returns:
        pending: Steps whose local or remote cleanup failed on this pass. Retry
            state remains in `checkpoints`; it is not persisted elsewhere.
    """
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
    """Save an evaluated policy and retain the best K checkpoints plus the latest.

    Call once per evaluation with increasing steps and a fixed `top_k`. These
    sequence requirements are caller obligations, not validated here.

    **Retention and failure behavior**

    - Rank higher scores first; ties favor later steps. Always keep this call's
      step. With `top_k == 0`, keep only that latest step.
    - Save locally before uploading. For an online W&B run, wait for upload
      completion before registering the record and pruning older versions.
    - Move the `latest` alias on every upload; add `best` when `top_k > 0` and this
      step ranks first. Use the same retained-step set locally and remotely.
    - Save/upload failures propagate without rolling back the newly written local
      file. Cleanup failures are logged and returned for retry on later calls.
      Retry records are in-memory only.

    Args:
        model: Policy to checkpoint, kept stable while saving.
        step: Current evaluation step used for ranking, dictionary keys, and filename.
        normalizer: Matching training normalization statistics passed to `save_policy`.
        flow_num_steps: Positive inference-step default passed to checkpoint saving.
        mean_reward: Finite score for this checkpoint; larger is better.
        top_k: Nonnegative integer number of highest-ranked steps to retain in
            addition to the latest when it is not already among them. Only the
            nonnegative bound is checked explicitly here.
        checkpoints: Mutable retention history. Successful saves add a record;
            successful pruning removes obsolete records.
        checkpoint_dir: Directory for `policy_step_<step>.pt`; saving creates parents.
        run: Online W&B run, or `None` for local-only saving.

    Returns:
        pending: Steps whose cleanup failed and remain tracked for future retries.

    Raises:
        ValueError: The supplied run is offline, `top_k` is negative, the score is
            nonfinite, or checkpoint validation fails.

    Filesystem, serialization, and upload errors propagate; cleanup errors are
    handled separately by [prune_checkpoints][hw1_imitation.evaluation.prune_checkpoints].
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
        scores, key=lambda saved_step: (-scores[saved_step], -saved_step)
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
    """Compute example-weighted validation loss without recording gradients.

    Temporarily call `eval()`, accumulate `batch_loss * batch_size`, and divide by
    the total example count. This assumes the policy returns a scalar batch mean.
    Input batches move to `device`; the model must already be placed compatibly.

    **Randomness and mode**

    - Reuse the supplied generator across batches. Stochastic policy losses consume
      it even in evaluation mode; deterministic losses may leave it unchanged.
    - To replay validation noise across checkpoints, supply a fresh generator with
      the same identity, batch order, and sampling shapes for each complete pass.
    - Restore the previous top-level training flag on normal or exceptional exit.
      This recursively sets child modes, not arbitrary preexisting mixed child modes.

    Args:
        model: Policy returning scalar mean loss from `compute_loss`.
        loader: Sized iterable of `(states, action_chunks)` batches with nonempty
            batches expected. No custom batch-shape validation is performed.
        device: Destination for validation tensors.
        generator: Caller-owned generator matching the sampling device. State is
            not restored after the pass, including a failed pass.
        show_progress: Enable a temporary progress bar when the terminal supports it.
        progress_position: Terminal row for the progress bar.

    Returns:
        results: Weighted loss and total processed example count. Loss finiteness
            is not separately validated.

    Raises:
        ValueError: The loader produces no examples.

    Policy and tensor-operation errors propagate without translation.
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
    env_stream: StreamId = StreamId.EVAL_ENV,
    policy_stream: StreamId = StreamId.EVAL_POLICY,
    env_variation: int = 0,
    policy_variation: int = 0,
    video_dir: Path | None = None,
    num_eval_episodes: int = NUM_EVAL_EPISODES,
    show_progress: bool = False,
    progress_position: int = 0,
) -> EvaluationResults:
    """Run Push-T rollouts and average their maximum observed rewards.

    **Rollout workflow**

    1. Create one state-observation Push-T environment and put the policy in eval mode.
    2. For each episode index, derive an environment seed and a fresh policy generator
       from the selected independent streams and variations.
    3. Normalize observations, sample without gradients, and denormalize actions.
       Check action shape, horizon, and finiteness, then clip to the Box bounds.
    4. Execute `chunk_size` actions before requesting another chunk, stopping at
       environment termination or truncation.

    The policy generator advances across chunks within an episode. Repeated calls
    with the same identities recreate those episode streams; replay also depends
    on unchanged environment/model behavior. The reported score is the mean of
    per-episode maxima, not a sum of rewards.

    **Ownership and cleanup**

    - Close the owned environment and restore the previous top-level model training
      flag on exit, including failures. Child modes are recursively set by `train`;
      mixed child-mode configurations are not individually restored.
    - Record frames after environment steps (not the initial reset), at 20 fps,
      to caller-owned paths `episode_<seven-digit index>.mp4`.
      Names repeat across calls; use a distinct directory to preserve earlier videos.
    - Add a video path only after rollout and writer closure succeed. Failures may
      leave partial files and propagate rather than return partial results.

    Environment, filesystem, encoding, and device errors propagate. Configuration
    checks below are not a comprehensive type validator; stream-coordinate validation
    is delegated to the factory and flow-step validation to the policy.

    Args:
        model: Policy producing normalized action chunks, already on `device`.
        normalizer: State/action statistics used for inference. No upfront statistics
            revalidation is performed by this helper.
        device: Inference tensor and policy-generator device.
        chunk_size: Positive number of actions executed before resampling. Must not
            exceed the predicted horizon, but may be smaller.
        video_size: `(width, height)` for recordings. Both must be positive and even
            when recording; ignored when recording is disabled.
        num_video_episodes: Nonnegative count of initial episodes to record, capped
            by the evaluation count. Zero disables recording.
        flow_num_steps: Integration steps forwarded to `sample_actions`; MSE ignores it.
        streams: Stateless factory for indexed episode seeds and generators.
        env_stream: Stream identity for environment reset seeds.
        policy_stream: Separate stream identity for policy sampling.
        env_variation: Realization coordinate for environment randomness.
        policy_variation: Realization coordinate for policy randomness.
        video_dir: Existing directory required when recording. Files remain owned
            by the caller after this function returns.
        num_eval_episodes: Positive integer number of episodes to run.
        show_progress: Enable a temporary progress bar on supported terminals.
        progress_position: Terminal row for the progress bar.

    Returns:
        results: Mean episode-maximum reward, completed episode count, and tuple of
            successfully completed video paths. Reward finiteness is not checked.

    Raises:
        ValueError: Episode/chunk counts are nonpositive, video count is negative,
            recording lacks a directory or valid dimensions, or predicted actions
            have incompatible shape, insufficient horizon, or nonfinite values.
            Invalid stream coordinates or flow steps can also fail in their owners.
        NotADirectoryError: The requested recording directory does not exist or is
            not a directory.
        TypeError: The environment's action space is not a Box or rendering does not
            return a NumPy array; delegated argument-type checks may also fail.
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
    # These are evaluated LIFO.
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
                stream=env_stream,
                variation=env_variation,
                index=episode_index,
            )

            policy_generator = streams.torch(
                stream=policy_stream,
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

            # Reached only after the rollout and writer closure succeeded.
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
