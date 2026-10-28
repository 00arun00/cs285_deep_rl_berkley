"""Run Push-T imitation training, checkpoint selection, and final evaluation.

**Run lifecycle**

1. Parse [TrainConfig][hw1_imitation.train.TrainConfig], load demonstrations,
   and split whole episodes into training and validation subsets.
2. Initialize a policy and training-only normalization, or load both from a
   checkpoint. Always start a fresh optimizer and training-step counter.
3. Train while reporting metrics, validating, and evaluating rollout scores.
   Retain checkpoints and optionally stop when improvement plateaus.
4. Reload the highest-scoring currently tracked checkpoint and evaluate it with
   separate final-evaluation streams and budget.

[run_training][hw1_imitation.train.run_training] owns the experiment directory,
logging contexts, and temporary checkpoint lifetime. This is a training driver,
not an optimizer/RNG-state resume mechanism.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import cast

import torch
from rich.console import Console
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import wandb
from hw1_imitation.checkpoint import load_policy, temp_checkpoint_dir
from hw1_imitation.config import TrainConfig, config_to_dict, parse_train_config
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
from hw1_imitation.model import BasePolicy, PolicyConfig, build_policy
from hw1_imitation.randomness import RandomStreamFactory, StreamId
from hw1_imitation.rng_modules import get_dropout_config, override_dropout_p

LOG_ROOT = "exp"


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
    """Build normalized action-window loaders over fixed episode subsets.

    **Loader behavior**

    - Training shuffles with `train_generator` and drops the final incomplete batch.
    - Validation preserves sample order and retains the final incomplete batch.
    - Both use zero worker processes. Their supplied generators manage loader RNG;
      fixed validation order does not imply its generator is never consumed.
    - Neither generator changes episode split membership. Retain the training
      generator across epochs to advance shuffles; recreate it to replay a run
      with unchanged data and loader settings.

    Args:
        config: Batch-size and action-padding settings. Architecture is supplied
            separately through `chunk_size` and the prepared data.
        train_episodes: Fixed training subset of whole episodes.
        validation_episodes: Fixed validation subset of whole episodes.
        normalizer: Shared statistics applied when constructing both window datasets.
        chunk_size: Action-window length, normally the actual policy horizon.
        train_generator: CPU generator retained by the shuffled training loader.
        validation_generator: CPU generator retained by the unshuffled validation loader.

    Returns:
        train_loader: Shuffled loader with `drop_last=True`.
        validation_loader: Ordered loader with `drop_last=False`.

    Raises:
        ValueError: A child dataset rejects empty inputs, dimensions or chunk counts,
            or PyTorch rejects loader settings.
        TypeError: A child rejects a non-integer or boolean chunk size.

    A valid window dataset can still yield zero training batches when every sample
    falls in the dropped remainder; `run_training` checks this separately.
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
    """Compute loss, backpropagate, and take one optimizer step.

    Set the model to training mode and clear gradients with `set_to_none=True`.
    Then call `compute_loss`, run backward, and update parameters. The caller must
    place the model and tensors on compatible devices. Mode, gradients, parameters,
    and consumed randomness are not restored on exit or failure.

    Args:
        model: Policy to train; expected to return a differentiable scalar loss.
        optimizer: Optimizer owning the policy parameters to update.
        state: State batch already placed on the model device.
        action_chunk: Target action windows in training coordinates on that device.
        generator: Caller-owned loss/dropout generator on the sampling device.

    Returns:
        loss: Detached loss tensor from before the parameter update, retaining its
            device. Detaching does not copy its storage.
    """
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
    """Execute one fresh training run and write its experiment outputs.

    **Data and initialization**

    - Use an available Torch accelerator, otherwise CPU. Download/load Push-T data
      and split whole episodes approximately 80/20 using the data-split stream.
    - For a fresh policy, fit normalization only on training episodes and initialize
      weights with the model-init stream. When loading a checkpoint, reuse its
      architecture and normalization; optionally override dropout.
    - Build window loaders using the actual model horizon. Start a new AdamW
      optimizer and step counter even when weights came from a checkpoint.

    **Reporting and selection**

    - Report training-window means, validation loss, and selection rollouts at their
      configured intervals and the last scheduled training step.
    - Reuse training loss/loader generators across batches and epochs. Restart the
      validation-loss generator for every complete validation pass. Selection
      rollouts recreate indexed episode streams; final rollouts use separate streams.
    - Save after each selection evaluation, retaining best K plus latest. Optional
      plateau stopping counts eligible selection evaluations, with warm-up converted
      from epochs to optimizer steps.
    - On scheduled completion or plateau, reload the best currently tracked record
      by `(score, step)` and run final evaluation. With latest-only retention this
      normally selects the latest, not the best checkpoint from the full history.
      Failed-cleanup records remain tracked and can refer to missing local files;
      loading such a selected record propagates the failure.
    - Flush any remaining training-metric window after leaving the loops. Plateau
      stopping does not force an additional off-schedule validation pass.

    **Output ownership**

    Create a new `exp/seed_<seed>_<timestamp>[_suffix]` directory relative to the
    working directory. Own the W&B and progress contexts; reject offline W&B runs.
    Disable W&B explicitly for local-only training. CSV settings do not disable
    checkpoint creation or the experiment directory. Final-evaluation videos
    remain local; the final metric call does not submit their paths to W&B.

    Persistent checkpoints stay under the experiment directory. When disabled,
    use temporary checkpoints for selection/final evaluation and remove them only
    on successful context exit. Failures propagate, close registered contexts, and
    may leave partial experiment outputs or temporary checkpoints for inspection.
    There is no transactional rollback or resumable optimizer/RNG-state snapshot.

    Args:
        config: Validated training settings. Loaded architecture overrides fresh-model
            options; current training, logging, and evaluation settings still apply.

    Raises:
        ValueError: No complete training batches remain, W&B is offline, or a
            delegated configuration/data/policy validation fails.
        FileExistsError: The generated experiment directory or a required fresh
            output destination already exists.

    Dataset, device, filesystem, W&B, and encoding errors propagate. The function
    returns no trained model; retained checkpoints and reports are its outputs.
    """
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

    if config.dropout_p is not None:
        override_dropout_p(model, config.dropout_p)
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
    wandb_run_config["training_config"] = {"dropout": get_dropout_config(model)}

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

        if run is not None and run.offline:
            raise ValueError(
                "Offline W&B is not supported by this training driver. "
                "Use --log-wandb False for local-only runs."
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
    """Parse process command-line options and run training.

    Delegate CLI handling to `parse_train_config` and execution to `run_training`.
    Help may exit through Tyro; training errors propagate to the invoking process.
    """
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
