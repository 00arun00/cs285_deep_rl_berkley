"""Format startup summaries, encode videos, and record experiment metrics.

**Output workflow**

1. Build a console table with
   [build_training_summary][hw1_imitation.logging_utils.build_training_summary].
2. Stream evaluation frames through
   [open_video_writer][hw1_imitation.logging_utils.open_video_writer].
3. Submit measurements and completed-file references through
   [ExperimentLogger][hw1_imitation.logging_utils.ExperimentLogger].

Video encoding and metric logging are independent. The caller owns experiment
directories, file retention, and W&B lifetime. Use one process per experiment
directory; concurrent writes and resuming existing metric files are unsupported.
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
from hw1_imitation.rng_modules import get_dropout_config

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
    """Build a Rich startup table from configuration and resolved runtime facts.

    Architecture and dropout come from the actual model, including loaded policies;
    training settings come from `config`. Paths are resolved for display without
    creating directories. This helper returns a table rather than printing it.
    It does not validate that supplied counts match the data or that paths exist.

    Args:
        config: Training settings controlling the displayed budget and options.
        model: Policy supplying actual architecture, trainable parameter count,
            and current per-layer dropout probabilities.
        run_name: Display name for the experiment.
        device: Human-readable device description.
        output_dir: Experiment output directory displayed as an absolute path.
        dataset_path: Dataset path displayed as an absolute path.
        train_episodes: Number of training episodes.
        validation_episodes: Number of validation episodes.
        train_samples: Number of training windows.
        validation_samples: Number of validation windows.
        steps_per_epoch: Optimizer steps per epoch, used to display the total budget.

    Returns:
        table: Newly constructed Rich table ready for the caller to render.
    """
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
    dropout = get_dropout_config(model)
    probabilities = set(dropout.values())
    if not dropout:
        dropout_description = "No dropout layers"
    elif len(probabilities) == 1:
        dropout_description = f"{next(iter(probabilities)):g} · training only"
    else:
        dropout_description = " · ".join(
            f"{name}: {p:g}" for name, p in dropout.items()
        )
        dropout_description += " · training only"
    table.add_row("Dropout", Text(dropout_description))
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
    """Yield an MP4 encoder and close it when the context exits.

    Supply consistent RGB uint8 frames with positive, even dimensions through
    `writer.append_data(frame)`. Encoding uses FFMPEG/libx264 with yuv420p and no
    macroblock-size adjustment; frame compatibility is delegated to the encoder.

    **File lifecycle**

    - Write directly to a new destination inside an existing directory.
    - Close the writer on context exit, including exceptions from the caller's body.
    - After normal closure, require a nonempty output file. This checks existence
      and size, not whether every frame is decodable.
    - Failures can leave partial files. Treat the path as successful only after
      normal context exit. The existence check is not a concurrent-writer lock.

    Args:
        path: New `.mp4` destination; extension matching is case-insensitive.
        fps: Positive playback frame rate. Only the positive bound is checked here.

    Yields:
        writer: ImageIO writer accepting frames through `append_data`. The caller
            borrows it only inside the context; this helper manages closure.

    Raises:
        ValueError: Frame rate is nonpositive or the extension is not `.mp4`.
        NotADirectoryError: The parent directory is missing or is not a directory.
        FileExistsError: The destination already exists.
        RuntimeError: Normal closure produced no file or an empty file.

    Filesystem, encoder, and caller-body errors propagate. The post-closure output
    check is skipped if the body or writer closure raises.
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
    """Write metric rows locally and mirror scalars to an optional W&B run.

    **Reporting contract**

    - Append one CSV row per logging call when local logging is enabled, then
      submit to W&B when a run is supplied. The operations are not atomic: remote
      failure can follow a successful local append.
    - Keep reporting single-process. There is no retry, deduplication, or resume
      support, and metric values are not validated for range or finiteness.
    - Record references to completed videos without acquiring deletion ownership.
      The caller also remains responsible for finishing the W&B run.

    Attributes:
        TRAIN_FIELDS: Ordered CSV schema for optimizer step, epoch, mean training
            loss over the reporting window, and contributing example count.
        VALIDATION_FIELDS: Ordered CSV schema for step, mean validation loss, and
            contributing example count.
        EVAL_FIELDS: Ordered CSV schema for step, rollout score, episode count,
            and JSON-encoded relative video paths.
        FINAL_EVAL_FIELDS: Ordered CSV schema for final evaluation, selected
            checkpoint metadata, and the CSV-only stop reason.
        log_csv: Whether to create metric CSVs and append local rows.
        directory: Resolved caller-owned experiment directory.
        train_path: Local `train.csv` destination.
        eval_path: Local `eval.csv` destination.
        validation_path: Local `validation.csv` destination.
        final_eval_path: Local `final_eval.csv` destination.
        run: Optional W&B run reference; this logger neither creates nor finishes it.
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
        """Prepare metric paths, optionally create CSVs, and define W&B metric axes.

        Require an existing directory even when CSV logging is disabled. When enabled,
        create each file exclusively and write its header; existing files are never
        intentionally overwritten. A later failure may leave earlier files created.

        If a run is provided, define `global_step` and the `train/*`, `validation/*`,
        `eval/*`, and `final/*` metric namespaces. Remote setup errors propagate.

        Args:
            directory: Caller-owned experiment directory. Metric files must not already
                exist when CSV logging is enabled.
            run: Active W&B run, or `None` to disable remote reporting.
            log_csv: Whether to create and append local metric files.

        Raises:
            NotADirectoryError: The resolved experiment directory is missing or not a directory.
            FileExistsError: A metric file already exists during exclusive creation.
            OSError: Resolving or creating local output fails.
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
        """Append one CSV row when local logging is enabled (implementation detail).

        Open and close the file on each call. Disabled logging returns before opening
        it. Missing row fields become empty CSV cells; extra fields are rejected.
        If the file was removed, append mode recreates it without a header.
        No value validation or rollback is added.

        Args:
            path: Destination opened in append mode; normally an initialized metric file.
            fields: Ordered field names used by `csv.DictWriter`.
            row: Values to serialize. Extra keys are rejected by the CSV writer.

        Raises:
            ValueError: The row contains keys outside `fields`.
            OSError: Opening or writing the file fails.
        """
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
        """Submit scalar metrics and video references (implementation detail).

        Do nothing when no run is supplied. Otherwise prefix metric names with
        `namespace`, retain a single unprefixed `global_step`, and attach videos as
        `<namespace>/rollout_ep<index>`. A `global_step` entry in `metrics` is ignored
        in favor of the explicit argument.

        Submission does not guarantee remote synchronization. W&B/media errors
        propagate; this helper neither validates video paths nor takes file ownership.

        Args:
            namespace: Metric prefix, normally `train`, `validation`, `eval`, or `final`.
            global_step: Step coordinate submitted as the common metric axis.
            metrics: Scalar values to prefix and submit.
            video_paths: Existing MP4 files wrapped as `wandb.Video` in sequence order.
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
        """Record a training-window summary in CSV and the `train` namespace.

        The caller computes the aggregate; this method does not accumulate batches.
        CSV append precedes remote submission under the class reporting contract.

        Args:
            global_step: Optimizer step at the end of the reporting window.
            epoch: Epoch index associated with the report.
            loss_window_mean: Caller-computed mean training loss over the window.
            window_examples_count: Number of examples contributing to that mean.
        """
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
        """Record a validation summary in CSV and the `validation` namespace.

        CSV append precedes remote submission; failures and retry limitations follow
        the class reporting contract.

        Args:
            global_step: Optimizer step associated with this validation pass.
            loss_mean: Caller-computed example-weighted mean validation loss.
            examples_count: Total examples contributing to the validation mean.
        """
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
        """Record final rollout results and the selected checkpoint's metadata.

        Append all fields to CSV, then send numeric metrics to the `final` namespace.
        `stop_reason` is CSV-only and is not included in the W&B payload.

        Args:
            global_step: Training step at which the final report is made.
            mean_reward: Final mean of per-episode maximum rewards.
            num_episodes: Number of episodes in final evaluation.
            checkpoint_step: Step of the selected checkpoint being evaluated.
            selection_mean_reward: Earlier score used to select that checkpoint.
            stop_reason: Caller-provided reason training ended; recorded as CSV text.
        """
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
        """Record rollout metrics and references to completed local videos.

        Resolve video paths and require each to be a file within the experiment
        directory after symlink resolution, even for W&B-only reporting. Video
        decodability is not checked. Store relative POSIX paths as a JSON
        array in CSV; pass resolved files to W&B. Files are not moved or deleted.

        When both reporting sinks are disabled, return without validating paths.
        Otherwise all path checks precede CSV append and subsequent W&B submission.

        Args:
            global_step: Optimizer step associated with this evaluation.
            mean_reward: Mean of per-episode maximum rewards, computed by the caller.
            num_episodes: Number of completed evaluation episodes.
            video_paths: Completed video files in report order. Relative paths resolve
                from the current working directory, not automatically from `directory`.

        Raises:
            FileNotFoundError: A resolved path is missing or is not a regular file.
            ValueError: A resolved video path lies outside the experiment directory.

        Filesystem and W&B/media errors propagate; a remote failure may follow a
        successful CSV append.
        """
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
