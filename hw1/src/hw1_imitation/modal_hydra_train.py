"""Train on Modal using packaged Hydra presets and key=value overrides."""

from pathlib import Path

import modal

from hw1_imitation.config_composition import compose_training_config
from hw1_imitation.modal_train import (
    DEFAULT_CPU,
    DEFAULT_GPU,
    VOLUME_PATH,
    env,
    image,
    volume,
)
from hw1_imitation.train import run_training

app = modal.App("hw1-imitation-hydra")


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 4,
    env=env,
    image=image,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
)
def train_remote(*args: str) -> None:
    config = compose_training_config(["data=modal", *args])
    run_training(config, output_root=Path(VOLUME_PATH) / "exp")
    volume.commit()
