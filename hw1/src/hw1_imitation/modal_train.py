"""Configure a Modal application for remote Push-T training.

**Declared runtime**

- Use the `hw1-imitation` application and named `hw1-imitation-volume`, mounted
  at `/vol` with creation requested if missing.
- Build a Debian-slim image with graphics libraries and `uv_sync` dependencies.
  Package the current working directory (`.`) at `/root/project`, using the
  simplified ignore rules from [load_gitignore_patterns][hw1_imitation.modal_train.load_gitignore_patterns].
- If `~/.netrc` exists where module setup runs, add it to the image at
  `/root/.netrc`. This is conditional file packaging, not a Modal secret lookup.
- Configure a T4 GPU, two CPUs, a four-hour timeout, `/root/project/src` as
  `PYTHONPATH`, and `/vol/wandb` as `WANDB_DIR`.

These image, volume, and application declarations run when the module is
imported. The decorated [train_remote][hw1_imitation.modal_train.train_remote]
entry point parses training options and uses `/vol/data` as its default dataset
location. The wrapper does not relocate the training driver's relative `exp/`
output directory into the volume.
"""

from pathlib import Path

import modal

from hw1_imitation.train import TrainConfig, parse_train_config, run_training


APP_NAME = "hw1-imitation"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
DEFAULT_GPU = "T4"
DEFAULT_CPU = 2.0
volume = modal.Volume.from_name("hw1-imitation-volume", create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    """Translate selected HW1 ignore entries into Modal directory-upload globs.

    Look for `.gitignore` in the HW1 project root, identified by
    `Path(__file__).resolve().parents[2]`. Return an empty list outside Modal's
    local setup context or when that file is absent.
    The upload itself packages `.`; callers should launch from the intended project
    working directory so this ignore file and the uploaded tree correspond.

    **Translation rules**

    - Strip surrounding whitespace and skip empty lines, comments, and negations
      beginning with `!`.
    - Remove leading `/`, so root anchoring is not preserved.
    - Convert trailing-directory entries to `**/<entry>/**`; otherwise prefix `**/`.

    This is a simplified translation, not full Git ignore semantics. Negated
    exceptions are discarded and anchored patterns become recursive.

    Returns:
        patterns: Ordered Modal ignore globs, or an empty list when no rules are read.

    Raises:
        OSError: Reading an existing ignore file fails.
        UnicodeError: Its contents cannot be decoded as UTF-8.
    """
    if not modal.is_local():
        return []

    root = Path(__file__).resolve().parents[2]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


# Build a container image with the project's dependencies using uv.
image = modal.Image.debian_slim().apt_install("libgl1", "libglib2.0-0").uv_sync()
if NETRC_PATH.is_file():
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )
image = image.add_local_dir(
    ".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)


app = modal.App(APP_NAME)

env = {
    "PYTHONPATH": f"{PROJECT_DIR}/src",
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",
}


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 4,
    env=env,
    image=image,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
)
def train_remote(*args: str) -> None:
    """Parse training options, execute remotely, and commit the mounted volume.

    Use `TrainConfig(data_dir=Path("/vol/data"))` as the parser defaults. Explicit
    options can override the data directory and all other supported training
    settings. Parsing uses these supplied arguments, not the remote process argv.

    Call `volume.commit()` only after `run_training` returns normally. Parse,
    training, or commit errors propagate; there is no explicit commit in a failure
    handler. Files outside the mounted `/vol` tree are not persisted by this call.

    Args:
        *args: Individual command-line option strings accepted by
            [parse_train_config][hw1_imitation.train.parse_train_config], for example
            `"--num-epochs", "1", "--no-log-wandb"`.

    Raises:
        SystemExit: Tyro handles help or rejects command-line options.

    Training validation/runtime errors and Modal volume errors propagate from their
    owners. This function returns no model or artifact object.
    """
    defaults = TrainConfig(data_dir=Path(VOLUME_PATH) / "data")
    config = parse_train_config(
        list(args),
        defaults=defaults,
        description="Train on Modal.",
    )
    run_training(config)
    volume.commit()
