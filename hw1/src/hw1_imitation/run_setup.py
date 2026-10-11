"""Prepare one run and call the trainer; shared by local and Modal launchers."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

from hw1_imitation.config import ExperimentConfig, config_to_dict


def launch_experiment(
    config: ExperimentConfig, *, overrides: Sequence[str] = ()
) -> None:
    """Save requested configuration and lineage before starting a new run.

    Explicit names must be unused. Generated names use system entropy without
    consuming training random streams. Relative paths use the invocation cwd.
    Failed runs keep their setup files for diagnosis.
    """
    started_at = datetime.now(timezone.utc)
    name = config.run.name or (
        f"seed_{config.train.randomness.seed}_"
        f"{started_at.strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}"
    )
    output_dir = (config.run.output_root / name).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(
        json.dumps(config_to_dict(config), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "overrides.json").write_text(
        json.dumps(list(overrides), indent=2) + "\n", encoding="utf-8"
    )

    training = config.train
    parent = None
    if training.warm_start is not None:
        source = training.warm_start.resolve()
        snapshot = output_dir / "initial_policy.pt"
        shutil.copyfile(source, snapshot)
        with snapshot.open("rb") as checkpoint:
            digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        parent = {
            "source": str(source),
            "snapshot": snapshot.name,
            "sha256": digest,
        }
        # Load the bytes we recorded, even if retention removes the source.
        training = replace(training, warm_start=snapshot)

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "started_at": started_at.isoformat(),
                "invocation_dir": str(Path.cwd()),
                "output_dir": str(output_dir),
                "parent_checkpoint": parent,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    from hw1_imitation.train import run_training

    run_training(training, output_dir=output_dir)
