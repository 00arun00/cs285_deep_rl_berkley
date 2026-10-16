# Homework 1: Imitation Learning

## Setup

This project uses `uv` for package management. `uv` is a Python package and environment manager from [Astral](https://astral.sh). It replaces tools like
`pip`, `pipx`, `conda`, and `virtualenv` with a single, simple interface. It is also much faster than prior tools.

### Installing `uv`

Run the following in your terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, open a new terminal so `uv` is on your `PATH`.

### Always use `uv run`

Do **not** run `python` or `pip` directly. Always run scripts through `uv run` so dependencies
and environments are handled automatically. If you want to add a new dependency, you can use `uv add`. This will add the dependency to `pyproject.toml`, update `uv.lock`, and install the package into your virtual environment.

Example:

```bash
uv run src/hw1_imitation/train.py --help
```

This should work out of the box with the provided starter code.

## Weights & Biases (wandb) login

These assignments use [Weights & Biases (WandB)](https://wandb.ai) for experiment tracking. WandB is a tool for logging and visualizing machine learning experiments. It is free for academic use. For W&B logging (enabled by default), log in using your API key. Local-only runs with `--no-log-wandb` do not require W&B login.

```bash
uv run wandb login
```

Follow the prompt to paste your API key.

## Using Modal

**Note that Modal is likely not necessary for this assignment. In testing, training was much faster on a local laptop CPU than on Modal. However, you may need to use Modal in future assignments, so if you want to get set up, here are the instructions:**

First, create a Modal account. You should recieve $30 in free credits, which will be plenty for this assignment. Then, you can train on Modal with the following command:

```bash
uv run modal run src/hw1_imitation/modal_train.py
```

This will build a Modal container and launch training remotely. You can pass the same flags as the local training script. If you are logged into WandB locally, your API key will be automatically forwarded to the Modal container.

Logs and checkpoints will be saved to a Modal volume called `hw1-imitation-volume`. To inspect the logs, you can use:

```bash
uv run modal volume ls hw1-imitation-volume exp
```

Then, you can download the logs and checkpoints to your local machine using a command like the following:

```bash
uv run modal volume get hw1-imitation-volume exp/<experiment_name>
```

## Dropout

- `--dropout-p 0.2` enables hidden-activation dropout for MSE and flow policies.
- When omitted, fresh models use `0` and loaded models retain their saved per-layer probabilities.
  An explicit value (including `0`) overrides every dropout layer; accepted range: `0 <= p < 1`.
- RNG-aware layers use the supplied training-loss generator; `.eval()` disables dropout.
- Checkpoints save `training_config.dropout`, mapping module names to probabilities.
  Loading requires matching dropout layer names and restores each value in evaluation
  mode.

## Evaluation episode counts

Use fewer episodes for periodic rollouts while keeping a full final evaluation:

```bash
uv run src/hw1_imitation/train.py --eval-episodes 5 --final-eval-episodes 100
```

| Evaluation | Episode budget | Policy |
| --- | --- | --- |
| Periodic and last training step | `--eval-episodes` | Current policy; used for checkpoint selection |
| After completion or early stopping | `--final-eval-episodes` | Best retained checkpoint; separate random streams |

Both budgets default to 100 and must be positive. Final scores do not affect
checkpoint selection. Results go to `final_eval.csv` when CSV logging is enabled
and `final/*` metrics when W&B is enabled.

## Reusing policy checkpoints

CSV metrics, W&B reporting, and persistent checkpoints are enabled by default:

```bash
uv run src/hw1_imitation/train.py --no-log-wandb
uv run src/hw1_imitation/train.py --no-log-csv --no-log-wandb --no-save-checkpoints
```

`--no-log-csv` prevents metric CSV creation and writes. `--no-log-wandb` skips
W&B initialization, metrics, video uploads, and checkpoint artifacts.
`--no-save-checkpoints` saves checkpoints in a unique temporary directory per
run instead of the experiment directory. W&B uploads and retention still apply.
Temporary checkpoint directories are deleted on success and retained on failure.
Progress, warnings, validation, and rollout evaluation remain enabled. Videos
are still recorded unless `--num-video-episodes 0` is supplied. The run directory
is still created with all three flags supplied.

Each selection evaluation (including the last training step) saves a checkpoint.
By default, its path is `exp/<experiment_name>/checkpoints/policy_step_<step>.pt`.
`--checkpoint-top-k` (default: 3) retains the best K by rollout mean reward plus
latest. Ties favor later checkpoints; zero keeps only latest. This applies to
both persistent and temporary storage.

With W&B enabled, checkpoints are also uploaded as model artifacts, with `best`
(when K > 0) and `latest` aliases. Pruning waits for upload completion. Local-only
runs apply the same retention without W&B. Cleanup failures are logged and retried
at the next save; pending cleanup is reported at the end of training. Retry state
does not persist across runs. Offline W&B runs remain unsupported; use
`--no-log-wandb` for local-only runs. W&B cache files and evaluation videos are
not covered by checkpoint retention.

The policy checkpoint format (version 1) contains:

- Model architecture, including the prediction horizon (`chunk_size`).
- Learned weights, saved as CPU tensors.
- Per-layer training configuration (`training_config.dropout`), restored on load.
- Observation and action normalization statistics.
- Default flow inference settings (`flow_num_steps`).

Files are written atomically and loaded with `weights_only=True`. They do not
contain optimizer history, random state, or training position. Legacy whole-model
`.pkl` files and earlier experimental resume-checkpoint schemas are unsupported.

### Load for inference or evaluation

The original training dataset is not needed:

```python
import torch
from hw1_imitation.checkpoint import load_policy

model, normalizer, inference_config = load_policy(
    "policy_step_12000.pt",
    device="cpu",
)

# raw_observations: NumPy array shaped (batch, state_dim).
states = torch.as_tensor(
    normalizer.normalize_state(raw_observations),
    dtype=torch.float32,
)
generator = torch.Generator(device=states.device).manual_seed(42)
with torch.no_grad():
    normalized_actions = model.sample_actions(
        states,
        generator=generator,
        num_steps=inference_config["flow_num_steps"],
    )
actions = normalizer.denormalize_action(normalized_actions.cpu().numpy())
# Clip actions to the environment's bounds before stepping it.
```

The loader returns the model in evaluation mode. For rollout evaluation, pass
`model.chunk_size`, the saved `flow_num_steps`, and a `RandomStreamFactory` via
`streams=` to `evaluate_policy()`. You can explicitly choose a different flow
integration step count when evaluating.

### Train further from saved weights

Run these commands from `hw1/`. To initialize a new training run from a policy:

```bash
uv run src/hw1_imitation/train.py --init-from /path/to/policy_step_12000.pt --num-epochs 20 --lr 0.0001
```

This trains for **20 additional epochs** with a fresh optimizer, seed, counters,
and logging run. Training and evaluation settings come from the new invocation's
flags and defaults, except dropout probabilities: each layer inherits its checkpoint value
unless `--dropout-p` uniformly overrides them. Use `--dropout-p 0` to disable saved dropout.
In particular, evaluations use the new run's `--flow-num-steps` value.

The loaded policy supplies its architecture and normalizer. `--policy-type`,
`--hidden-dims`, and `--chunk-size` apply only to fresh models. Data preparation and
rollout evaluation use the loaded model's actual horizon. The actual architecture
is recorded under `model_config` in W&B alongside the requested run settings.
New demonstrations must have compatible observation/action dimensions and meaning;
the saved normalization is reused rather than fitted again.

Omit `--init-from` to train a new model. Training loss is logged every
`log_interval` steps and at the final step; loss windows may span epochs.
This workflow does not attempt to reproduce an interrupted training trajectory.

### Checks

GitHub Actions runs the HW1 checks on every pull request irrespective of the
target branch and every push to `main`, using Linux and the Python
version in `.python-version`.
The `HW1 checks` job installs locked dependencies, runs Ruff lint and test
formatting checks, and runs pytest. The accelerator test skips when no accelerator
is available. The job is required by the `main` branch merge rules.

```bash
uv sync --locked --dev
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check tests
uv run src/hw1_imitation/train.py --help
uv run pytest -v
```

All tests use pytest, with parameterized cases for policy variants, output
combinations, and invalid inputs.
To run only the flow checkpoint variants, use
`uv run pytest tests/test_checkpoint.py -k flow -v`.

Tests cover policy round trips, training, evaluation budgets, logging, checkpoint
retention, and random-stream isolation. Data access, physics environments, and W&B
services are mocked. An accelerator transfer test runs when hardware is available;
the suite does not exercise GPU training or remote uploads.

## Early stopping

Stop when rollout reward plateaus. Disabled by default.

| Setting | Default | Purpose |
| --- | --- | --- |
| `early_stopping_patience` | `None` | Enable with a positive number of stale evaluations |
| `early_stopping_min_epochs` | `100` | Warm-up before counting stale evaluations |
| `early_stopping_min_delta` | `0.005` | Reward gain must exceed this to reset patience |
