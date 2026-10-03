"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
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
from hw1_imitation.evaluation import (
    compute_validation_loss,
    evaluate_policy,
    log_checkpoint_artifact,
)
from hw1_imitation.logging_utils import ExperimentLogger
from hw1_imitation.model import BasePolicy, PolicyConfig, PolicyType, build_policy

LOGDIR_PREFIX = "exp"


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
    data_split_seed: int = 42
    pad_action_chunk_with_last_action: bool = True

    # Training — applies to both fresh and loaded policies.
    seed: int = 42
    num_epochs: int = 400
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0

    # Validation and rollout evaluation, measured in training steps.
    validation_interval: int = 100
    eval_interval: int = 10_000

    # Rollout inference and video recording.
    # flow_num_steps is ignored by MSE policies.
    flow_num_steps: int = 10
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)

    # Training metrics and experiment tracking.
    log_interval: int = 100
    wandb_project: str = "cs285-hw1-imitation-learning"
    exp_name: str | None = None

    def __post_init__(self):
        for name in (
            "num_epochs",
            "batch_size",
            "chunk_size",
            "log_interval",
            "validation_interval",
            "eval_interval",
            "flow_num_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def build_loaders(
    config: TrainConfig,
    train_episodes: Dataset[Episode],
    validation_episodes: Dataset[Episode],
    normalizer: Normalizer,
    *,
    chunk_size: int,
) -> tuple[DataLoader, DataLoader]:
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
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, validation_loader


def train_step(
    model: BasePolicy,
    optimizer: torch.optim.Optimizer,
    state: torch.Tensor,
    action_chunk: torch.Tensor,
) -> torch.Tensor:
    """Run 1 step of training"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model.compute_loss(state=state, action_chunk=action_chunk)
    loss.backward()
    optimizer.step()
    return loss.detach()


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)

    device = torch.accelerator.current_accelerator()
    if device is None or not torch.accelerator.is_available():
        device = torch.device("cpu")

    zarr_path = download_pusht(config.data_dir)
    episodes_dataset = load_episodes_dataset(zarr_path=zarr_path)

    # We are not spliting out a test becuase we use evaluate_policy
    # where we have access to new gym env states to test performance against
    train_episodes_subset, validation_episodes_subset = random_split(
        dataset=episodes_dataset,
        lengths=[0.8, 0.2],
        generator=torch.Generator().manual_seed(config.data_split_seed),
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
            )
        ).to(device=device)

    else:
        model, normalizer, _ = load_policy(path=config.init_from, device=device)

    train_loader, validation_loader = build_loaders(
        config=config,
        train_episodes=train_episodes_subset,
        validation_episodes=validation_episodes_subset,
        normalizer=normalizer,
        chunk_size=model.chunk_size,
    )

    if len(train_loader) == 0:
        raise ValueError(
            "No training batches; reduces batch_size or check the dataset."
        )

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
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    log_dir.mkdir(parents=True, exist_ok=False)

    total_steps = config.num_epochs * len(train_loader)
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

    with wandb.init(
        project=config.wandb_project,
        config=wandb_run_config,
        name=exp_name,
        dir=str(log_dir),
    ) as run:
        logger = ExperimentLogger(log_dir, run=run)

        with tqdm(
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
        ) as progress:
            for epoch in range(config.num_epochs):
                display_metrics["epoch"] = f"{epoch + 1}/{config.num_epochs}"
                progress.set_postfix(display_metrics, refresh=False)
                for state, action_chunk in train_loader:
                    loss = train_step(
                        model=model,
                        optimizer=optimizer,
                        state=state.to(device),
                        action_chunk=action_chunk.to(device),
                    )
                    global_step += 1

                    # compute_loss() returns a batch mean. Weight it by the
                    # number of examples before combining it with other batches.

                    batch_examples = state.shape[0]
                    loss_sum += loss.item() * batch_examples
                    window_examples_count += batch_examples

                    training_complete = global_step == total_steps
                    eval_due = (
                        global_step % config.eval_interval == 0 or training_complete
                    )
                    log_due = (
                        global_step % config.log_interval == 0 or training_complete
                    )
                    validation_due = (
                        global_step % config.validation_interval == 0
                        or training_complete
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
                        validation_result = compute_validation_loss(
                            model=model,
                            loader=validation_loader,
                            device=device,
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
                            flow_num_steps=config.flow_num_steps,
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

                        display_metrics["eval"] = (
                            f"{result.mean_reward:.3f}@{global_step}"
                        )
                        progress.set_postfix(display_metrics, refresh=False)
                        progress.set_description_str("Saving")

                        log_checkpoint_artifact(
                            model=model,
                            step=global_step,
                            normalizer=normalizer,
                            flow_num_steps=config.flow_num_steps,
                        )

                        progress.set_description_str("Train")

            progress.set_description_str("Done")


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
