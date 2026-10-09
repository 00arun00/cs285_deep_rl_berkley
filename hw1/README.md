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

## Evaluation episode counts

Use fewer episodes for periodic rollouts while keeping a full final evaluation:

```bash
uv run src/hw1_imitation/train.py --eval-episodes 5 --final-eval-episodes 100
```

Both counts must be positive and default to 100. If a periodic evaluation lands
on the final training step, only the final evaluation runs, using
`--final-eval-episodes`. Evaluation intervals, video recording, and checkpoint
schedules are unchanged. Video count is capped by the rollout episode count.

## Reusing policy checkpoints

CSV metrics, W&B reporting, and checkpoint saving are enabled by default and
can be disabled independently:

```bash
uv run src/hw1_imitation/train.py --no-log-wandb
uv run src/hw1_imitation/train.py --no-log-csv --no-log-wandb --no-save-checkpoints
```

`--no-log-csv` prevents metric CSV creation and writes. `--no-log-wandb` skips
W&B initialization, metrics, video uploads, and checkpoint artifacts.
`--no-save-checkpoints` skips both local checkpoints and their W&B artifacts.
Progress, warnings, validation, and rollout evaluation remain enabled. Videos
are still recorded unless `--num-video-episodes 0` is supplied. The run directory
is still created even when all three outputs are disabled.

When checkpoint saving is enabled, each rollout evaluation (including the final
training step) saves `exp/<experiment_name>/checkpoints/policy_step_<step>.pt`.
`--checkpoint-top-k` (default: 3) retains the best K by rollout mean reward plus
latest. Ties favor earlier checkpoints; zero keeps only latest. This setting has
no effect when checkpoint saving is disabled.

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
flags and defaults; the previous training configuration is not inherited.
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

```bash
uv run src/hw1_imitation/train.py --help
uv run python -m unittest discover -s tests -v
```

Tests cover policy round trips and further training on CPU with mocked rollout
evaluation and W&B services. They do not exercise GPU training or remote uploads.
