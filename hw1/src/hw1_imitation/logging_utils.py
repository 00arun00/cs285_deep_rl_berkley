"""Format console summaries, write local outputs, and report to W&B.

Video encoding and metric logging are independent operations. Evaluation
uses ``open_video_writer`` to create files; ``ExperimentLogger`` records
measurements and references to files that already exist.

The caller owns experiment directories, file retention, and W&B lifetime.
One process must own each experiment directory. Resume and concurrent
writes are not supported.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v2 as imageio
from imageio.core.format import Format
from rich import box
from rich.table import Table
from rich.text import Text

import wandb

if TYPE_CHECKING:
    from hw1_imitation.model import BasePolicy
    from hw1_imitation.train import TrainConfig

Scalar = int | float
RecordValue = Scalar | str | wandb.Video


def build_training_summary(
    config: TrainConfig,
    *,
    model: BasePolicy,
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
    architecture = model.config
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
        if config.init_from is None
        else "loaded weights · fresh optimizer"
    )
    table.add_row("Policy", f"{architecture.policy_type.upper()} · {initialization}")
    if config.init_from is not None:
        table.add_row("Initial checkpoint", Text(str(config.init_from.resolve())))
    table.add_row(
        "Hidden layers",
        " → ".join(map(str, architecture.hidden_dims)) or "None",
    )
    table.add_row(
        "Observation/action",
        f"{architecture.state_dim} / {architecture.action_dim} dimensions",
    )
    table.add_row("Action chunk", f"{architecture.chunk_size} steps")
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    table.add_row("Parameters", f"{parameters:,} trainable")
    if architecture.policy_type == "flow":
        table.add_row("Flow sampling", f"{config.flow_num_steps} steps")

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
        if config.pad_action_chunk_with_last_action
        else "Disabled · full chunks only",
    )
    table.add_row(
        "Normalization",
        "Computed from training episodes"
        if config.init_from is None
        else "Loaded from checkpoint",
    )
    table.add_section()
    table.add_row("Root seed", str(config.seed))
    table.add_row("Data split variation", str(config.data_split_variation))
    table.add_row(
        "Training variations",
        f"Initialization: {config.model_init_variation} · "
        f"loader: {config.train_loader_variation} · "
        f"loss: {config.train_loss_variation}",
    )
    table.add_row(
        "Validation variations",
        f"Loader: {config.validation_loader_variation} · "
        f"loss: {config.validation_loss_variation}",
    )
    table.add_row(
        "Evaluation variations",
        f"Environment: {config.eval_env_variation} · "
        f"policy: {config.eval_policy_variation}",
    )

    table.add_section()

    table.add_row(
        "Optimizer",
        f"AdamW · lr {config.lr:g} · weight decay {config.weight_decay:g}",
    )
    table.add_row("Dropout", f"{config.dropout_p:g} · training only")
    table.add_row(
        "Training budget",
        f"{config.num_epochs:,} epochs · {config.num_epochs * steps_per_epoch:,} optimizer steps",
    )
    table.add_row(
        "Batch size", f"{config.batch_size:,} · incomplete training batch dropped"
    )
    table.add_row("Metric interval", f"Every {config.log_interval:,} steps + final")
    table.add_row(
        "Early stopping",
        "Disabled"
        if config.early_stopping_patience is None
        else (
            f"{config.early_stopping_patience} stale rollouts; "
            f"warm-up {config.early_stopping_min_epochs} epochs; "
            f"minimum gain {config.early_stopping_min_delta:g}"
        ),
    )
    table.add_row("Validation", f"Every {config.validation_interval:,} steps + final")
    table.add_row(
        "Rollout evaluation",
        f"Every {config.eval_interval:,} steps · {config.eval_episodes} episodes",
    )
    table.add_row("Final evaluation", f"{config.final_eval_episodes} episodes")
    table.add_section()

    table.add_row(
        "CSV logging",
        Text(
            "Enabled" if config.log_csv else "Disabled",
            style="green" if config.log_csv else "dim",
        ),
    )
    table.add_row(
        "W&B",
        Text(
            f"Enabled · {config.wandb_project}" if config.log_wandb else "Disabled",
            style="green" if config.log_wandb else "dim",
        ),
    )
    retention = (
        f"Best {config.checkpoint_top_k} + latest"
        if config.checkpoint_top_k
        else "Latest only"
    )
    storage = (
        "Experiment directory" if config.save_checkpoints else "Temporary directory"
    )
    table.add_row(
        "Checkpoints",
        Text(
            f"{retention} · after each evaluation · to {storage}",
            style="" if config.save_checkpoints else "dim",
        ),
    )
    width, height = config.video_size
    table.add_row(
        "Videos",
        Text(
            f"Up to {config.num_video_episodes} episodes/evaluation · {width} × {height}"
            if config.num_video_episodes
            else "Disabled",
            style="" if config.num_video_episodes else "dim",
        ),
    )
    return table


@contextmanager
def open_video_writer(
    path: Path,
    *,
    fps: int = 20,
) -> Iterator[Format.Writer]:
    """Stream frames into a new MP4 file and close its encoder on exit.

    The caller supplies RGB uint8 frames with consistent, even dimensions
    through ``writer.append_data(frame)``.

    Frames are written directly to the destination. A failure or process
    interruption may leave a partial file. Callers must record the path
    as successful only after this context exits normally.

    Args:
        path: New MP4 file inside an existing caller-owned directory.
        fps: Positive playback frame rate.

    Raises:
        ValueError: The frame rate or filename extension is invalid.
        NotADirectoryError: The parent directory does not exist.
        FileExistsError: The destination already exists.
        RuntimeError: Encoding produced no output data.

    Filesystem and encoder errors propagate. The existence check assumes
    one writer; it is not a concurrency lock.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if path.suffix.lower() != ".mp4":
        raise ValueError("Video path must have an .mp4 extension")
    if not path.parent.is_dir():
        raise NotADirectoryError(path.parent)
    if path.exists():
        raise FileExistsError(path)

    writer = imageio.get_writer(
        str(path),
        # ImageIO accepts a format name; its stub incorrectly requires Type(Format).
        format="FFMPEG",  # pyright: ignore[reportArgumentType]
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    # Keep the Writer type; ImageIO's __enter__ stub returns its base class.
    with writer:
        yield writer

    # This checks for missing output, not whether every frame is decodable.
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Video writer produced no output: {path}")


class ExperimentLogger:
    """Write scalar metrics locally and mirror them to an optional W&B run.

    Each logging call appends one CSV row when log_csv is enabled.
    If a W&B run is supplied, the same scalar values are also submitted to the run.

    CSV writes happen prior to W&B submissions.
    These operations are not atomic.
    i.e. a W&B failure can occur after the CSV row has been written.

    The logger is intended for periodic reporting from a single training process.
    It does not provide retry, deduplication or resume behavior.
    """

    TRAIN_FIELDS: tuple[str, ...] = (
        "global_step",
        "epoch",
        "loss_window_mean",
        "window_examples_count",
    )
    VALIDATION_FIELDS: tuple[str, ...] = (
        "global_step",
        "loss_mean",
        "examples_count",
    )
    EVAL_FIELDS: tuple[str, ...] = (
        "global_step",
        "mean_reward",
        "num_episodes",
        "video_paths",
    )
    FINAL_EVAL_FIELDS: tuple[str, ...] = (
        "global_step",
        "mean_reward",
        "num_episodes",
        "checkpoint_step",
        "selection_mean_reward",
        "stop_reason",
    )

    def __init__(
        self,
        directory: Path,
        run: wandb.Run | None = None,
        *,
        log_csv: bool = True,
    ) -> None:
        """Create metric files in an existing experiment directory.

        Args:
            directory: Caller-owned experiment directory. When CSV logging is
                enabled, metric CSV files must not already exist.
            run: Active W&B run, or None to disable W&B reporting.
            log_csv: Create and append local metric CSV files.
        The caller remains responsible for finishing the run.

        Initialization errors propagate and may leave partially created
        files. Existing records are never intentionally overwritten.
        """
        self.log_csv = log_csv
        self.directory = directory.resolve()
        if not self.directory.is_dir():
            raise NotADirectoryError(self.directory)

        self.train_path = self.directory / "train.csv"
        self.eval_path = self.directory / "eval.csv"
        self.validation_path = self.directory / "validation.csv"
        self.final_eval_path = self.directory / "final_eval.csv"

        if self.log_csv:
            for path, fields in (
                (self.train_path, self.TRAIN_FIELDS),
                (self.validation_path, self.VALIDATION_FIELDS),
                (self.eval_path, self.EVAL_FIELDS),
                (self.final_eval_path, self.FINAL_EVAL_FIELDS),
            ):
                with path.open("x", encoding="utf-8", newline="") as file:
                    csv.DictWriter(file, fieldnames=fields).writeheader()

        self.run = run

        if self.run is not None:
            self.run.define_metric("global_step")
            self.run.define_metric("train/*", step_metric="global_step")
            self.run.define_metric("validation/*", step_metric="global_step")
            self.run.define_metric("eval/*", step_metric="global_step")
            self.run.define_metric("final/*", step_metric="global_step")

    def _append_row(
        self,
        path: Path,
        fields: tuple[str, ...],
        row: Mapping[str, RecordValue],
    ) -> None:
        """Append a CSV row when local metric logging is enabled."""
        if not self.log_csv:
            return

        with path.open("a", encoding="utf-8", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writerow(row)

    def _submit_to_wandb(
        self,
        namespace: str,
        global_step: int,
        metrics: Mapping[str, Scalar],
        video_paths: Sequence[Path] = (),
    ) -> None:
        """Submit metrics and existing videos without taking file ownership.

        Successful submission does not guarantee remote synchronization.

        NOTE: This is mostly a dumb utility function
        """
        if self.run is None:
            return

        payload: dict[str, RecordValue] = {
            "global_step": global_step,
        }
        payload.update(
            {
                f"{namespace}/{name}": value
                for name, value in metrics.items()
                if name != "global_step"
            }
        )

        for index, path in enumerate(video_paths):
            payload[f"{namespace}/rollout_ep{index}"] = wandb.Video(
                str(path),
                format="mp4",
            )

        self.run.log(payload)

    def log_train(
        self,
        *,
        global_step: int,
        epoch: int,
        loss_window_mean: float,
        window_examples_count: int,
    ) -> None:
        """Record training metrics"""
        metrics = {
            "global_step": global_step,
            "epoch": epoch,
            "loss_window_mean": loss_window_mean,
            "window_examples_count": window_examples_count,
        }
        self._append_row(self.train_path, self.TRAIN_FIELDS, metrics)
        self._submit_to_wandb(
            namespace="train", global_step=global_step, metrics=metrics
        )

    def log_validation(
        self,
        *,
        global_step: int,
        loss_mean: float,
        examples_count: int,
    ) -> None:
        metrics = {
            "global_step": global_step,
            "loss_mean": loss_mean,
            "examples_count": examples_count,
        }

        self._append_row(
            self.validation_path,
            self.VALIDATION_FIELDS,
            metrics,
        )
        self._submit_to_wandb(
            namespace="validation",
            global_step=global_step,
            metrics=metrics,
        )

    def log_final_evaluation(
        self,
        *,
        global_step: int,
        mean_reward: float,
        num_episodes: int,
        checkpoint_step: int,
        selection_mean_reward: float,
        stop_reason: str,
    ) -> None:
        """Record final evaluation metrics."""

        metrics: dict[str, Scalar] = {
            "global_step": global_step,
            "mean_reward": mean_reward,
            "num_episodes": num_episodes,
            "checkpoint_step": checkpoint_step,
            "selection_mean_reward": selection_mean_reward,
        }

        row = {**metrics, "stop_reason": stop_reason}

        self._append_row(
            path=self.final_eval_path,
            fields=self.FINAL_EVAL_FIELDS,
            row=row,
        )

        self._submit_to_wandb(
            namespace="final",
            global_step=global_step,
            metrics=metrics,
        )

    def log_eval(
        self,
        *,
        global_step: int,
        mean_reward: float,
        num_episodes: int,
        video_paths: Sequence[Path] = (),
    ) -> None:
        """Record evaluation metrics and completed local video references."""

        if not self.log_csv and self.run is None:
            return

        resolved_paths = tuple(path.resolve() for path in video_paths)
        relative_paths: list[str] = []

        for path in resolved_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            relative_paths.append(path.relative_to(self.directory).as_posix())

        metrics = {
            "global_step": global_step,
            "mean_reward": mean_reward,
            "num_episodes": num_episodes,
        }

        row = {**metrics, "video_paths": json.dumps(relative_paths)}
        self._append_row(
            path=self.eval_path,
            fields=self.EVAL_FIELDS,
            row=row,
        )
        self._submit_to_wandb(
            namespace="eval",
            global_step=global_step,
            metrics=metrics,
            video_paths=resolved_paths,
        )
