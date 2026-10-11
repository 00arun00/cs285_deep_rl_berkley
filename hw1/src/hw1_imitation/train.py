"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from rich.console import Console
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import wandb
from hw1_imitation.checkpoint import load_policy
from hw1_imitation.config import DataConfig, TrainConfig, config_to_dict
from hw1_imitation.data import (
    Episode,
    EpisodeChunkDataset,
    Normalizer,
    download_pusht,
    load_episodes_dataset,
)
from hw1_imitation.evaluation import (
    CheckpointRecord,
    compute_validation_loss,
    evaluate_policy,
    save_checkpoint_and_retain,
)
from hw1_imitation.logging_utils import ExperimentLogger
from hw1_imitation.model import BasePolicy, PolicyConfig, build_policy
from hw1_imitation.randomness import RandomStreamFactory, StreamId
from hw1_imitation.reporting import build_training_summary


def build_loaders(
    data_config: DataConfig,
    train_episodes: Dataset[Episode],
    validation_episodes: Dataset[Episode],
    normalizer: Normalizer,
    *,
    chunk_size: int,
    batch_size: int,
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
        pad_action_chunk=data_config.pad_action_chunk_with_last_action,
    )
    validation_dataset = EpisodeChunkDataset(
        episodes=validation_episodes,
        chunk_length=chunk_size,
        normalizer=normalizer,
        pad_action_chunk=data_config.pad_action_chunk_with_last_action,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
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


def run_training(config: TrainConfig, *, output_dir: Path) -> None:
    """Train in an existing directory prepared by the caller."""
    if not output_dir.is_dir():
        raise ValueError("output_dir must be an existing directory")
    streams = RandomStreamFactory(root_seed=config.randomness.seed)

    device = torch.accelerator.current_accelerator()
    if device is None or not torch.accelerator.is_available():
        device = torch.device("cpu")

    zarr_path = download_pusht(config.data.dir)
    episodes_dataset = load_episodes_dataset(zarr_path=zarr_path)

    # We are not splitting out a test because policy performance is
    # evaluated through environment rollouts using separately controlled
    # evaluation randomness.
    train_episodes_subset, validation_episodes_subset = random_split(
        dataset=episodes_dataset,
        lengths=[0.8, 0.2],
        generator=streams.torch(
            StreamId.DATA_SPLIT,
            variation=config.randomness.data_split_variation,
        ),
    )

    if config.warm_start is None:
        normalizer = Normalizer.from_episodes_data(episodes=train_episodes_subset)

        model = build_policy(
            PolicyConfig(
                policy_type=config.model.policy_type,
                state_dim=normalizer.state_dim,
                action_dim=normalizer.action_dim,
                chunk_size=config.model.chunk_size,
                hidden_dims=config.model.hidden_dims,
            ),
            cpu_generator=streams.torch(
                StreamId.MODEL_INIT,
                variation=config.randomness.model_init_variation,
                device="cpu",
            ),
            target_device=device,
        )

    else:
        model, normalizer, _ = load_policy(
            path=config.warm_start,
            device=device,
        )

    device = next(model.parameters()).device

    # Create fresh for each training run; advance across its batches and epochs.
    # Streams with matching identities on the same backend reproduce the starting RNG state.
    train_loss_generator = streams.torch(
        StreamId.TRAIN_LOSS,
        variation=config.randomness.train_loss_variation,
        device=device,
    )

    train_loader, validation_loader = build_loaders(
        data_config=config.data,
        batch_size=config.optimization.batch_size,
        train_episodes=train_episodes_subset,
        validation_episodes=validation_episodes_subset,
        normalizer=normalizer,
        chunk_size=model.chunk_size,
        train_generator=streams.torch(
            StreamId.TRAIN_LOADER,
            variation=config.randomness.train_loader_variation,
        ),
        validation_generator=streams.torch(
            StreamId.VALIDATION_LOADER,
            variation=config.randomness.validation_loader_variation,
        ),
    )

    if len(train_loader) == 0:
        raise ValueError("No training batches; reduce batch_size or check the dataset.")

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.optimization.lr,
        weight_decay=config.optimization.weight_decay,
    )
    global_step = 0

    log_dir = output_dir
    exp_name = output_dir.name
    (output_dir / "run_facts.json").write_text(
        json.dumps(
            {
                "model": asdict(model.config),
                "device": str(device),
                "dataset_path": str(zarr_path.resolve()),
                "normalization": "checkpoint" if config.warm_start else "training_data",
                "train_episodes": len(train_episodes_subset),
                "validation_episodes": len(validation_episodes_subset),
                "steps_per_epoch": len(train_loader),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    total_steps = config.optimization.num_epochs * len(train_loader)
    loss_sum = 0.0
    window_examples_count = 0

    display_metrics = {
        "epoch": f"1/{config.optimization.num_epochs}",
        "loss": "—",
        "val": "—",
        "eval": "—",
    }

    wandb_run_config = config_to_dict(config)
    wandb_run_config["model_config"] = asdict(model.config)

    tracking_context = (
        wandb.init(
            project=config.logging.project,
            config=wandb_run_config,
            name=exp_name,
            dir=str(log_dir),
        )
        if config.logging.wandb
        else nullcontext(None)
    )

    with tracking_context as run:
        if run is not None and run.offline:
            raise ValueError(
                "Offline W&B is not supported by this training driver. "
                "Use train.logging.wandb=false for local-only runs."
            )

        logger = ExperimentLogger(log_dir, run=run, log_csv=config.logging.csv)
        checkpoints: dict[int, CheckpointRecord] = {}
        pending_cleanup: set[int] = set()

        if config.logging.show_summary:
            model_device = next(model.parameters()).device
            device_label = str(model_device)
            if model_device.type == "cuda":
                device_label += f" · {torch.cuda.get_device_name(model_device)}"
            Console(stderr=True).print(
                build_training_summary(
                    config=config,
                    architecture=model.config,
                    trainable_parameters=sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
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
            for epoch in range(config.optimization.num_epochs):
                display_metrics["epoch"] = (
                    f"{epoch + 1}/{config.optimization.num_epochs}"
                )
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
                    eval_due = (
                        global_step % config.evaluation.interval == 0
                        or training_complete
                    )
                    log_due = (
                        global_step % config.logging.interval == 0 or training_complete
                    )
                    validation_due = (
                        global_step % config.evaluation.validation_interval == 0
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
                        # Restart validation RNG for each complete pass so checkpoints use
                        # matching draws, assuming unchanged batch order and sampling shapes.
                        validation_result = compute_validation_loss(
                            model=model,
                            loader=validation_loader,
                            device=device,
                            generator=streams.torch(
                                StreamId.VALIDATION_LOSS,
                                variation=config.randomness.validation_loss_variation,
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
                        if config.evaluation.num_video_episodes > 0:
                            video_dir = log_dir / "videos" / f"step_{global_step:08d}"
                            video_dir.mkdir(parents=True, exist_ok=False)

                        result = evaluate_policy(
                            model=model,
                            normalizer=normalizer,
                            device=device,
                            chunk_size=model.chunk_size,
                            video_size=config.evaluation.video_size,
                            num_video_episodes=config.evaluation.num_video_episodes,
                            num_eval_episodes=(
                                config.evaluation.final_episodes
                                if training_complete
                                else config.evaluation.episodes
                            ),
                            flow_num_steps=config.evaluation.flow_num_steps,
                            streams=streams,
                            env_variation=config.randomness.eval_env_variation,
                            policy_variation=config.randomness.eval_policy_variation,
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
                        if config.checkpoints.enabled:
                            progress.set_description_str("Saving")

                            pending_cleanup = save_checkpoint_and_retain(
                                model=model,
                                step=global_step,
                                checkpoint_dir=log_dir / "checkpoints",
                                run=run,
                                mean_reward=result.mean_reward,
                                top_k=config.checkpoints.top_k,
                                checkpoints=checkpoints,
                                normalizer=normalizer,
                                flow_num_steps=config.evaluation.flow_num_steps,
                            )

                        progress.set_description_str("Train")

            progress.set_description_str("Done")

        if pending_cleanup:
            logging.warning(
                "Training finished with checkpoint cleanup pending for steps %s. "
                "Extra local files or W&B artifacts may remain.",
                sorted(pending_cleanup),
            )
