# Set up the Push-T development environment

Prepare the environment in `hw1/` before running the [small local test](test-runs.md) or [automated checks](tests-and-checks.md).

## Prerequisites

- A local checkout of this repository.
- A terminal with `uv` available on its `PATH`; installation is described below.
- Network access for initial Python and dependency installation. A later training run may also download the Push-T dataset.

HW1 declares Python **3.11 or newer**; its `.python-version` selects **3.11**, which CI also uses. A GPU is optional: the trainer uses an available PyTorch accelerator, falling back to CPU.

## Install uv if needed

On macOS or Linux, use the command from the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This downloads and executes Astral's installer, which installs `uv` and may update your shell setup. Open a new terminal afterward and check `uv --version`. Skip installation if `uv` is already available.

## Install the locked environment

From the repository root:

```bash
cd hw1
uv --version
uv python install
uv sync --locked --dev
```

- `uv python install` obtains the Python version selected by `.python-version` when needed.
- `uv sync --locked --dev` creates or updates the project environment using `uv.lock`, including development dependencies. It fails if the lockfile needs updating.
- These setup commands may download packages and write environment/cache files. They do not start training or download demonstrations.

Keep subsequent HW1 commands in **`hw1/`**. Use `uv run` so commands use this project's environment; `--no-sync` reuses the installed environment without synchronizing dependencies again.

## Verify the setup

Run from `hw1/`:

```bash
uv run --no-sync python --version
uv run --no-sync src/hw1_imitation/train.py --help
uv run --no-sync pytest tests/test_checkpoint.py::test_cli_preserves_config_defaults -v
```

Expected results:

1. Python reports a version compatible with the project (normally 3.11 with the checked-in version file).
2. Trainer help lists options such as `--policy-type`, `--num-epochs`, and logging controls, without starting training.
3. The named CLI-defaults test passes. This checks argument parsing against the configuration defaults; it does not establish that a real training run succeeds.

For the complete lint and test sequence, continue to [Tests and checks](tests-and-checks.md).

See [training CLI syntax](../reference/training.md#command-line-syntax) for explicit
boolean values and tuple arguments. To search across configurations after setup,
follow [Run a hyperparameter sweep](guides/run-sweep.md).

## Know which tools you need

| Tool | Role | When you need it |
| --- | --- | --- |
| `uv` | Python and dependency management. | Setup and project commands. |
| PyTorch and the runtime dependencies | Policies, data handling, and Push-T evaluation. | Installed by the sync command. |
| Ruff | Lint and formatting checks. | Before submitting code changes; included in project dependencies. |
| pytest | Run the existing test suite. | Development; installed with `--dev`. |
| Hypothesis | Generate examples for property-based tests. | Development; installed with `--dev` and used by pytest. |
| JupyterLab | Interactive exploration. | Optional; included in `--dev`. Start with `uv run --no-sync jupyter lab` from `hw1/`. |
| Weights & Biases | Experiment reporting and artifact uploads. | Optional service; the small-run recipe disables it. |
| Modal | Remote execution. | Optional service; local development and tests do not require a Modal account. |

The W&B and Modal Python packages are installed as runtime dependencies even when their services are unused. Ordinary training enables W&B by default; use the [small-run recipe](test-runs.md) for a local run without login or uploads.

## Enable W&B reporting only when needed

For a run that should upload metrics and artifacts, authenticate from `hw1/`:

```bash
uv run --no-sync wandb login
```

Follow the prompt for your W&B API key. This contacts the service and saves authentication for later runs. Skip login for the [small local test](test-runs.md), which supplies `--log-wandb False`.

## Keep documentation tooling separate

The repository-root environment builds the site; it does not replace the `hw1/` environment. Return to the root before following [documentation authoring](../contributing/index.md).

## Common setup mistakes

| Symptom | Check |
| --- | --- |
| `uv` is not found | Open a new terminal after installation and confirm `uv --version` works. |
| Trainer script or `tests/` cannot be found | Confirm the working directory is `hw1/`, not the repository root. |
| A module or test tool is missing | Run `uv sync --locked --dev` in `hw1/`, then retry through `uv run --no-sync`. |
| Locked synchronization fails | Check that `pyproject.toml` and `uv.lock` belong to the same checkout. Resolve the dependency mismatch before retrying. |
| W&B asks for credentials during a local test | Use every flag in the small-run recipe, including `--log-wandb False`. |

For persistent environment, device, or CI differences, see [Troubleshooting](troubleshooting.md).
