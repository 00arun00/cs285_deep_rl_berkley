"""Human-readable presentation helpers for training runs.

Build display objects from validated configuration and resolved runtime facts.
Callers own when and where to render them. Keep formatting here and run
execution, metric persistence, and output lifetimes in their owning modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich import box
from rich.table import Table
from rich.text import Text

from hw1_imitation.config import TrainConfig

if TYPE_CHECKING:
    from hw1_imitation.model import PolicyConfig


def build_training_summary(
    config: TrainConfig,
    *,
    architecture: PolicyConfig,
    trainable_parameters: int,
    run_name: str,
    device: str,
    output_dir: Path,
    dataset_path: Path,
    train_episodes: int,
    validation_episodes: int,
    train_samples: int,
    validation_samples: int,
    steps_per_epoch: int,
) -> Table:
    """Format TrainConfig and resolved runtime facts for the startup display."""
    table = Table(
        title="Push-T · Training summary",
        title_style="bold",
        title_justify="center",
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 1),
    )
    table.add_column("Setting", style="cyan")
    table.add_column("Value", overflow="fold", ratio=1)

    table.add_row("Run", Text(run_name))
    table.add_row("Device", Text(device))
    table.add_row("Output directory", Text(str(output_dir.resolve())))
    table.add_section()

    initialization = (
        "initialized from scratch"
        if config.warm_start is None
        else "loaded weights · fresh optimizer"
    )
    table.add_row("Policy", f"{architecture.policy_type.upper()} · {initialization}")
    if config.warm_start is not None:
        table.add_row("Initial checkpoint", Text(str(config.warm_start.resolve())))
    table.add_row(
        "Hidden layers",
        " → ".join(map(str, architecture.hidden_dims)) or "None",
    )
    table.add_row(
        "Observation/action",
        f"{architecture.state_dim} / {architecture.action_dim} dimensions",
    )
    table.add_row("Action chunk", f"{architecture.chunk_size} steps")
    table.add_row("Parameters", f"{trainable_parameters:,} trainable")
    if architecture.policy_type == "flow":
        table.add_row("Flow sampling", f"{config.evaluation.flow_num_steps} steps")

    table.add_section()

    table.add_row("Dataset", Text(str(dataset_path.resolve())))
    table.add_row(
        "Episodes",
        f"{train_episodes:,} train / {validation_episodes:,} validation",
    )
    table.add_row("Training samples", f"{train_samples:,}")
    table.add_row("Validation samples", f"{validation_samples:,}")
    table.add_row(
        "Chunk padding",
        "Repeat last action"
        if config.data.pad_action_chunk_with_last_action
        else "Disabled · full chunks only",
    )
    table.add_row(
        "Normalization",
        "Computed from training episodes"
        if config.warm_start is None
        else "Loaded from checkpoint",
    )
    table.add_section()
    table.add_row("Root seed", str(config.randomness.seed))
    table.add_row("Data split variation", str(config.randomness.data_split_variation))
    table.add_row(
        "Training variations",
        f"Initialization: {config.randomness.model_init_variation} · "
        f"loader: {config.randomness.train_loader_variation} · "
        f"loss: {config.randomness.train_loss_variation}",
    )
    table.add_row(
        "Validation variations",
        f"Loader: {config.randomness.validation_loader_variation} · "
        f"loss: {config.randomness.validation_loss_variation}",
    )
    table.add_row(
        "Evaluation variations",
        f"Environment: {config.randomness.eval_env_variation} · "
        f"policy: {config.randomness.eval_policy_variation}",
    )

    table.add_section()

    table.add_row(
        "Optimizer",
        f"AdamW · lr {config.optimization.lr:g} · weight decay {config.optimization.weight_decay:g}",
    )
    table.add_row(
        "Training budget",
        f"{config.optimization.num_epochs:,} epochs · {config.optimization.num_epochs * steps_per_epoch:,} optimizer steps",
    )
    table.add_row(
        "Batch size",
        f"{config.optimization.batch_size:,} · incomplete training batch dropped",
    )
    table.add_row("Metric interval", f"Every {config.logging.interval:,} steps + final")
    table.add_row(
        "Validation", f"Every {config.evaluation.validation_interval:,} steps + final"
    )
    table.add_row(
        "Rollout evaluation",
        f"Every {config.evaluation.interval:,} steps · {config.evaluation.episodes} episodes",
    )
    table.add_row("Final evaluation", f"{config.evaluation.final_episodes} episodes")
    table.add_section()

    table.add_row(
        "CSV logging",
        Text(
            "Enabled" if config.logging.csv else "Disabled",
            style="green" if config.logging.csv else "dim",
        ),
    )
    table.add_row(
        "W&B",
        Text(
            f"Enabled · {config.logging.project}"
            if config.logging.wandb
            else "Disabled",
            style="green" if config.logging.wandb else "dim",
        ),
    )
    retention = (
        f"Best {config.checkpoints.top_k} + latest"
        if config.checkpoints.top_k
        else "Latest only"
    )
    table.add_row(
        "Checkpoints",
        Text(
            f"{retention} · after each evaluation"
            if config.checkpoints.enabled
            else "Disabled",
            style="" if config.checkpoints.enabled else "dim",
        ),
    )
    width, height = config.evaluation.video_size
    table.add_row(
        "Videos",
        Text(
            f"Up to {config.evaluation.num_video_episodes} episodes/evaluation · {width} × {height}"
            if config.evaluation.num_video_episodes
            else "Disabled",
            style="" if config.evaluation.num_video_episodes else "dim",
        ),
    )
    return table
