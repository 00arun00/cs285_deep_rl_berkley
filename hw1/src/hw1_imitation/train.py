"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
import tyro
from rich.console import Console
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import wandb
from hw1_imitation.checkpoint import load_policy
from hw1_imitation.data import (
    Episode,
    EpisodeChunkDataset,
    Normalizer,
    download_pusht,
    load_episodes_dataset,
)
from hw1_imitation.early_stopping import EarlyStopping
from hw1_imitation.evaluation import (
    CheckpointRecord,
    compute_validation_loss,
    evaluate_policy,
    save_checkpoint_and_retain,
)
from hw1_imitation.logging_utils import ExperimentLogger, build_training_summary
from hw1_imitation.model import BasePolicy, PolicyConfig, PolicyType, build_policy
from hw1_imitation.randomness import RandomStreamFactory, StreamId

LOG_ROOT = "exp"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    # Policy initialization.
    # A loaded policy supplies its own architecture and normalizer.
    init_from: Path | None = None

    # Model architecture — used only when init_from is None.
    policy_type: PolicyType = "mse"
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    chunk_size: int = 8

    # Dataset and sample preparation.
    data_dir: Path = Path("data")
    pad_action_chunk_with_last_action: bool = True

    # Training — applies to both fresh and loaded policies.
    num_epochs: int = 400
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0

    # Validation and rollout evaluation, measured in training steps.
    validation_interval: int = 100
    eval_interval: int = 10_000
    eval_episodes: int = 100
    final_eval_episodes: int = 100

    # None disables plateau stopping. Warm-up does not count toward patience.
    early_stopping_patience: int | None = None
    early_stopping_min_epochs: int = 100
    early_stopping_min_delta: float = 0.005

    # Retain the best K rollout checkpoints, plus latest (0 = latest only).
    save_checkpoints: bool = True
    checkpoint_top_k: int = 3

    # Rollout inference and video recording.
    # flow_num_steps is ignored by MSE policies.
    flow_num_steps: int = 10
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)

    # Root seed identifies the experiment family.
    # Keep it fixed when varying individual sources of randomness.
    seed: int = 42

    # Independently selectable realizations of each stream.
    data_split_variation: int = 0
    model_init_variation: int = 0
    train_loader_variation: int = 0
    train_loss_variation: int = 0
    validation_loader_variation: int = 0
    validation_loss_variation: int = 0
    eval_env_variation: int = 0
    eval_policy_variation: int = 0
    final_eval_env_variation: int = 0
    final_eval_policy_variation: int = 0

    # Training metrics and experiment tracking.
    log_csv: bool = True
    log_wandb: bool = True
    log_interval: int = 100
    wandb_project: str = "cs285-hw1-imitation-learning"
    exp_name: str | None = None

    # Print the effective configuration and dataset/model facts before training.
    show_summary: bool = True

    def __post_init__(self) -> None:
        for name in (
            "num_epochs",
            "batch_size",
            "chunk_size",
            "log_interval",
            "validation_interval",
            "eval_interval",
            "eval_episodes",
            "final_eval_episodes",
            "flow_num_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        streams = RandomStreamFactory(root_seed=self.seed)
        for stream, variation in (
            (StreamId.DATA_SPLIT, self.data_split_variation),
            (StreamId.MODEL_INIT, self.model_init_variation),
            (StreamId.TRAIN_LOADER, self.train_loader_variation),
            (StreamId.TRAIN_LOSS, self.train_loss_variation),
            (StreamId.VALIDATION_LOADER, self.validation_loader_variation),
            (StreamId.VALIDATION_LOSS, self.validation_loss_variation),
            (StreamId.EVAL_ENV, self.eval_env_variation),
            (StreamId.EVAL_POLICY, self.eval_policy_variation),
        ):
            streams.validate(stream, variation=variation)

        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_min_epochs < 0:
            raise ValueError("early_stopping_min_epochs must be nonnegative")
        if (
            not math.isfinite(self.early_stopping_min_delta)
            or self.early_stopping_min_delta < 0
        ):
            raise ValueError("early_stopping_min_delta must be finite and nonnegative")

        if self.checkpoint_top_k < 0:
            raise ValueError("checkpoint_top_k must be nonnegative")

        if self.num_video_episodes < 0:
            raise ValueError("num_video_episodes must be nonnegative")

        if self.num_video_episodes > 0:
            width, height = self.video_size
            if width <= 0 or height <= 0 or width % 2 or height % 2:
                raise ValueError("Video dimensions must be positive and even")


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


# Needed only to preserve the temp dir with checkpoints in case of runtime failure.
# This directory will still be in /tmp so will get auto cleaned by os; based on os config.
# But gives us the opportunity to inspect and potentially continue with
# further training, if a training run fails mid training.
@contextmanager
def temp_checkpoint_dir(exp_name: str):
    path = Path(tempfile.mkdtemp(prefix=f"hw1-{exp_name}-checkpoints-"))
    yield path
    # Only clean up after successful completion.
    shutil.rmtree(path)


def build_loaders(
    config: TrainConfig,
    train_episodes: Dataset[Episode],
    validation_episodes: Dataset[Episode],
    normalizer: Normalizer,
    *,
    chunk_size: int,
    train_generator: torch.Generator,
    validation_generator: torch.Generator,
) -> tuple[DataLoader, DataLoader]:
    """Build loaders over fixed train and validation subsets.

    The training generator advances to produce successive epoch shuffles.
    Recreating it with the same identity at the start of another run reproduces
    that shuffle sequence, assuming unchanged dataset and loader settings.

    Validation sample order is fixed. Neither generator changes split membership.
    """
    train_dataset = EpisodeChunkDataset(
        episodes=train_episodes,
        chunk_length=chunk_size,
        normalizer=normalizer,
        pad_action_chunk=config.pad_action_chunk_with_last_action,
    )
    validation_dataset = EpisodeChunkDataset(
        episodes=validation_episodes,
        chunk_length=chunk_size,
        normalizer=normalizer,
        pad_action_chunk=config.pad_action_chunk_with_last_action,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        generator=validation_generator,
    )

    return train_loader, validation_loader


def train_step(
    model: BasePolicy,
    optimizer: torch.optim.Optimizer,
    state: torch.Tensor,
    action_chunk: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Run 1 step of training"""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss = model.compute_loss(
        state=state,
        action_chunk=action_chunk,
        generator=generator,
    )

    loss.backward()
    optimizer.step()
    return loss.detach()


def run_training(config: TrainConfig) -> None:
    streams = RandomStreamFactory(root_seed=config.seed)

    device = torch.accelerator.current_accelerator()
    if device is None or not torch.accelerator.is_available():
        device = torch.device("cpu")

    zarr_path = download_pusht(config.data_dir)
    episodes_dataset = load_episodes_dataset(zarr_path=zarr_path)

    # We are not splitting out a test because policy performance is
    # evaluated through environment rollouts using separately controlled
    # evaluation randomness.
    train_episodes_subset, validation_episodes_subset = random_split(
        dataset=episodes_dataset,
        lengths=[0.8, 0.2],
        generator=streams.torch(
            StreamId.DATA_SPLIT,
            variation=config.data_split_variation,
        ),
    )

    if config.init_from is None:
        normalizer = Normalizer.from_episodes_data(episodes=train_episodes_subset)

        model = build_policy(
            PolicyConfig(
                policy_type=config.policy_type,
                state_dim=normalizer.state_dim,
                action_dim=normalizer.action_dim,
                chunk_size=config.chunk_size,
                hidden_dims=config.hidden_dims,
            ),
            cpu_generator=streams.torch(
                StreamId.MODEL_INIT,
                variation=config.model_init_variation,
                device="cpu",
            ),
            target_device=device,
        )

    else:
        model, normalizer, _ = load_policy(
            path=config.init_from,
            device=device,
        )

    device = next(model.parameters()).device

    # Create fresh for each training run; advance across its batches and epochs.
    # Streams with matching identities on the same backend reproduce the starting RNG state.
    train_loss_generator = streams.torch(
        StreamId.TRAIN_LOSS,
        variation=config.train_loss_variation,
        device=device,
    )

    train_loader, validation_loader = build_loaders(
        config=config,
        train_episodes=train_episodes_subset,
        validation_episodes=validation_episodes_subset,
        normalizer=normalizer,
        chunk_size=model.chunk_size,
        train_generator=streams.torch(
            StreamId.TRAIN_LOADER,
            variation=config.train_loader_variation,
        ),
        validation_generator=streams.torch(
            StreamId.VALIDATION_LOADER,
            variation=config.validation_loader_variation,
        ),
    )

    if len(train_loader) == 0:
        raise ValueError("No training batches; reduce batch_size or check the dataset.")

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    global_step = 0

    # Setup logging
    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"

    # The training driver owns the directory and all output lifetimes.
    log_dir = Path(LOG_ROOT) / exp_name
    log_dir.mkdir(parents=True, exist_ok=False)

    total_steps = config.num_epochs * len(train_loader)
    stopper = (
        EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            start_step=config.early_stopping_min_epochs * len(train_loader),
            objective="max",
        )
        if config.early_stopping_patience is not None
        else None
    )
    stop_requested = False
    stop_reason = "max_epochs"
    loss_sum = 0.0
    window_examples_count = 0

    display_metrics = {
        "epoch": f"1/{config.num_epochs}",
        "loss": "—",
        "val": "—",
        "eval": "—",
    }

    wandb_run_config = config_to_dict(config)
    wandb_run_config["model_config"] = asdict(model.config)

    with ExitStack() as stack:
        # Checkpoint Directory management.
        # Auto clears temp dirs on successful completion only.
        checkpoint_dir = (
            log_dir / "checkpoints"
            if config.save_checkpoints
            else stack.enter_context(temp_checkpoint_dir(exp_name=exp_name))
        )

        run = (
            stack.enter_context(
                wandb.init(
                    project=config.wandb_project,
                    config=wandb_run_config,
                    name=exp_name,
                    dir=str(log_dir),
                )
            )
            if config.log_wandb
            else None
        )

        progress = stack.enter_context(
            tqdm(
                total=total_steps,
                desc="Train",
                unit="step",
                position=0,
                leave=True,
                dynamic_ncols=True,
                mininterval=0.5,
                miniters=1,
                disable=None,
                postfix=display_metrics,
            )
        )

        if run is not None and run.offline:
            raise ValueError(
                "Offline W&B is not supported by this training driver. "
                "Use --no-log-wandb for local-only runs."
            )

        logger = ExperimentLogger(log_dir, run=run, log_csv=config.log_csv)
        checkpoints: dict[int, CheckpointRecord] = {}
        pending_cleanup: set[int] = set()

        if config.show_summary:
            model_device = next(model.parameters()).device
            device_label = str(model_device)
            if model_device.type == "cuda":
                device_label += f" · {torch.cuda.get_device_name(model_device)}"
            Console(stderr=True).print(
                build_training_summary(
                    config=config,
                    model=model,
                    run_name=exp_name,
                    device=device_label,
                    output_dir=log_dir,
                    dataset_path=zarr_path,
                    train_episodes=len(train_episodes_subset),
                    validation_episodes=len(validation_episodes_subset),
                    train_samples=len(
                        cast(Sequence, train_loader.dataset)
                    ),  # can be safely cast here since dataset does have len.
                    validation_samples=len(
                        cast(Sequence, validation_loader.dataset)
                    ),  # can be safely cast here since dataset does have len.
                    steps_per_epoch=len(train_loader),
                )
            )

        epoch = 0
        for epoch in range(config.num_epochs):
            display_metrics["epoch"] = f"{epoch + 1}/{config.num_epochs}"
            progress.set_postfix(display_metrics, refresh=False)
            for state, action_chunk in train_loader:
                loss = train_step(
                    model=model,
                    optimizer=optimizer,
                    state=state.to(device),
                    action_chunk=action_chunk.to(device),
                    generator=train_loss_generator,
                )
                global_step += 1

                # compute_loss() returns a batch mean. Weight it by the
                # number of examples before combining it with other batches.

                batch_examples = state.shape[0]
                loss_sum += loss.item() * batch_examples
                window_examples_count += batch_examples

                training_complete = global_step == total_steps
                eval_due = global_step % config.eval_interval == 0 or training_complete
                log_due = global_step % config.log_interval == 0 or training_complete
                validation_due = (
                    global_step % config.validation_interval == 0 or training_complete
                )

                if log_due:
                    mean_loss = loss_sum / window_examples_count

                    logger.log_train(
                        global_step=global_step,
                        epoch=epoch,
                        loss_window_mean=mean_loss,
                        window_examples_count=window_examples_count,
                    )

                    display_metrics["loss"] = f"{mean_loss:.4g}"
                    progress.set_postfix(display_metrics, refresh=False)

                    loss_sum = 0.0
                    window_examples_count = 0

                # Advance after the optimizer step; tqdm throttles redraws.
                progress.update(1)

                if validation_due:
                    # Restart validation RNG for each complete pass so checkpoints use
                    # matching draws, assuming unchanged batch order and sampling shapes.
                    validation_result = compute_validation_loss(
                        model=model,
                        loader=validation_loader,
                        device=device,
                        generator=streams.torch(
                            StreamId.VALIDATION_LOSS,
                            variation=config.validation_loss_variation,
                            device=device,
                        ),
                    )
                    logger.log_validation(
                        global_step=global_step,
                        loss_mean=validation_result.loss_mean,
                        examples_count=validation_result.examples_count,
                    )

                    display_metrics["val"] = f"{validation_result.loss_mean:.4g}"
                    progress.set_postfix(display_metrics, refresh=False)

                if eval_due:
                    progress.set_description_str("Evaluating")
                    video_dir = None
                    if config.num_video_episodes > 0:
                        video_dir = log_dir / "videos" / f"step_{global_step:08d}"
                        video_dir.mkdir(parents=True, exist_ok=False)

                    result = evaluate_policy(
                        model=model,
                        normalizer=normalizer,
                        device=device,
                        chunk_size=model.chunk_size,
                        video_size=config.video_size,
                        num_video_episodes=config.num_video_episodes,
                        num_eval_episodes=config.eval_episodes,
                        flow_num_steps=config.flow_num_steps,
                        streams=streams,
                        env_stream=StreamId.EVAL_ENV,
                        policy_stream=StreamId.EVAL_POLICY,
                        env_variation=config.eval_env_variation,
                        policy_variation=config.eval_policy_variation,
                        video_dir=video_dir,
                        show_progress=not progress.disable,
                        progress_position=1,
                    )

                    logger.log_eval(
                        global_step=global_step,
                        mean_reward=result.mean_reward,
                        num_episodes=result.num_episodes,
                        video_paths=result.video_paths,
                    )

                    display_metrics["eval"] = f"{result.mean_reward:.3f}@{global_step}"
                    progress.set_postfix(display_metrics, refresh=False)

                    progress.set_description_str("Saving")

                    pending_cleanup = save_checkpoint_and_retain(
                        model=model,
                        step=global_step,
                        checkpoint_dir=checkpoint_dir,
                        run=run,
                        mean_reward=result.mean_reward,
                        top_k=config.checkpoint_top_k,
                        checkpoints=checkpoints,
                        normalizer=normalizer,
                        flow_num_steps=config.flow_num_steps,
                    )

                    if stopper is not None and not training_complete:
                        stop_requested = stopper.update(
                            step=global_step,
                            value=result.mean_reward,
                        )
                        if stop_requested:
                            stop_reason = "plateau"
                    progress.set_description_str("Train")

                final_eval_due = training_complete or stop_requested

                if final_eval_due:
                    progress.set_description_str("Running Final Evaluation")

                    # Retrieve best model / weights.
                    # NOTE: we tie break to last step; [nit] double descent.
                    best_step = max(
                        checkpoints, key=lambda step: (checkpoints[step].score, step)
                    )
                    best_model_record = checkpoints[best_step]
                    best_model, best_normalizer, inference = load_policy(
                        path=best_model_record.path, device=device
                    )

                    video_dir = None
                    if config.num_video_episodes > 0:
                        # TODO: confirm that the best_step here is in the same scale as global_step; not epochs.
                        video_dir = log_dir / "videos" / f"best_step_{best_step:08d}"
                        video_dir.mkdir(parents=True, exist_ok=False)

                    # call final evaluation using evaluate_policy()
                    final_eval_result = evaluate_policy(
                        model=best_model,
                        normalizer=best_normalizer,
                        device=device,
                        chunk_size=best_model.chunk_size,
                        video_size=config.video_size,
                        num_video_episodes=config.num_video_episodes,
                        num_eval_episodes=config.final_eval_episodes,
                        flow_num_steps=inference["flow_num_steps"],
                        streams=streams,
                        env_stream=StreamId.FINAL_EVAL_ENV,
                        policy_stream=StreamId.FINAL_EVAL_POLICY,
                        env_variation=config.final_eval_env_variation,
                        policy_variation=config.final_eval_policy_variation,
                        video_dir=video_dir,
                        show_progress=not progress.disable,
                        progress_position=1,
                    )

                    logger.log_final_evaluation(
                        global_step=global_step,
                        mean_reward=final_eval_result.mean_reward,
                        num_episodes=final_eval_result.num_episodes,
                        checkpoint_step=best_step,
                        selection_mean_reward=best_model_record.score,
                        stop_reason=stop_reason,
                    )

                if stop_requested:
                    break
            if stop_requested:
                break

        if window_examples_count:
            logger.log_train(
                global_step=global_step,
                epoch=epoch,
                loss_window_mean=loss_sum / window_examples_count,
                window_examples_count=window_examples_count,
            )
        progress.set_description_str("Stopped (plateau)" if stop_requested else "Done")

        logging.info("Training ended at step %s: %s", global_step, stop_reason)
        if run is not None:
            run.summary["stop_reason"] = stop_reason

        if pending_cleanup:
            logging.warning(
                "Training finished with checkpoint cleanup pending for steps %s. "
                "Extra local files or W&B artifacts may remain.",
                sorted(pending_cleanup),
            )


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
