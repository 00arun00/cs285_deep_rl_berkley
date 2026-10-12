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

## Training command syntax

Individual runs and W&B sweep trials use the same `hw1_imitation.train` entry
point and explicit-value Tyro parser. From `hw1`, view all options and defaults:

```bash
uv run python -m hw1_imitation.train --help
```

For example:

```bash
uv run python -m hw1_imitation.train \
  --policy-type flow \
  --hidden-dims "(256, 256, 256)" \
  --pad-action-chunk-with-last-action False \
  --lr 0.0003 \
  --log-wandb False
```

| Kind | Example | Notes |
| --- | --- | --- |
| Boolean | `--log-csv False` or `--log-csv=True` | Explicit `True`/`False`; bare flags, `--no-log-csv`, and `0`/`1` are not accepted. |
| Tuple | `--hidden-dims "(256, 256, 256)"` | Quote one Python tuple literal; lists and separate numeric arguments are not accepted. A singleton is `"(256,)"`. |
| Fixed-size tuple | `--video-size "(256, 256)"` | Exactly two integer dimensions. |
| Number | `--lr 1e-4` or `--batch-size=128` | Integer fields require integers. |
| Optional value | `--init-from None` | Use literal `None`, not `null`. |
| Path/string | `--data-dir "data with spaces"` | Quote values containing spaces. |

Omitted arguments retain their defaults. Hyphens and underscores are accepted in
option names; ordinary scalar values accept both space-separated and equals
forms. `--help` shows the generated accepted syntax, types, and defaults.
The same conventions apply to training flags passed through Modal, which shares
this parser. For example, old `--no-log-wandb` commands become
`--log-wandb False`, and old space-separated architecture values become a quoted
tuple. Configuration and model constraints still apply after argument parsing.

## Regularization

Two training-time regularizers are available for both fresh and loaded policies:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--weight-decay` | `0.0` | Decoupled weight decay inside AdamW. |
| `--dropout-p` | `0.0` | Dropout probability applied to hidden activations during training only. `0` disables it and draws nothing. |
| `--state-noise-std` | `0.0` | Standard deviation of Gaussian noise added to normalized training states. `0` disables it. |

For example:

```bash
uv run python -m hw1_imitation.train \
  --dropout-p 0.2 \
  --state-noise-std 0.03 \
  --weight-decay 0.0001 \
  --log-wandb False
```

Dropout masks and state noise draw from their own caller-owned random streams
(`TRAIN_LOSS` and `TRAIN_STATE_NOISE`), so runs replay exactly for a fixed
`--seed` and per-stream `--*-variation` values. Dropout applies only in
training mode: validation loss and rollout evaluation run under `model.eval()`,
which disables it even when a probability is configured. The startup summary
and the W&B configuration both record the effective values.

The sweep files search `dropout_p` with a uniform distribution over `[0.0, 0.4]`
and `state_noise_std` log-uniformly over `[0.001, 0.1]` in addition to the
existing `weight_decay` range. The log-uniform floor of `0.001` normalized
units is negligible, so the search can still effectively disable noise; there
is no need for an exact zero arm.

## Weights & Biases (wandb) login

These assignments use [Weights & Biases (WandB)](https://wandb.ai) for experiment tracking. WandB is a tool for logging and visualizing machine learning experiments. It is free for academic use. For W&B logging (enabled by default), log in using your API key. Local-only runs with `--log-wandb False` do not require W&B login.

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
uv run src/hw1_imitation/train.py --log-wandb False
uv run src/hw1_imitation/train.py --log-csv False --log-wandb False --save-checkpoints False
```

`--log-csv False` prevents metric CSV creation and writes. `--log-wandb False` skips
W&B initialization, metrics, video uploads, and checkpoint artifacts.
`--save-checkpoints False` skips both local checkpoints and their W&B artifacts.
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
`--log-wandb False` for local-only runs. W&B cache files and evaluation videos are
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

## Local W&B sweeps

The MSE and flow search spaces live in `sweeps/mse.yaml` and
`sweeps/flow.yaml`. They use Bayesian search and inherit omitted trainer defaults
(including batch size 128, seed 42, and 100 periodic evaluation episodes).

Each trial invokes `hw1_imitation.train`, the same entry point used for individual
runs. W&B expands `${args}` into explicit values such as
`--pad_action_chunk_with_last_action=False`. Architecture choices use quoted
tuple strings, for example `"(256, 256, 256)"`. Unspecified fields inherit trainer
defaults; the resolved configuration is logged to W&B.

Sweeps registered with a previous wrapper or JSON command must be replaced:
finish or stop their agents, then register the updated YAML files to obtain new
sweep IDs. Editing local YAML does not update an existing registered command.

From `hw1`, register each sweep once:

```bash
uv run wandb sweep sweeps/mse.yaml
uv run wandb sweep sweeps/flow.yaml
```

Use the sweep IDs printed by those commands to run sequential local trials:

```bash
uv run wandb agent --count 5 ENTITY/PROJECT/MSE_SWEEP_ID
uv run wandb agent --count 5 ENTITY/PROJECT/FLOW_SWEEP_ID
```

Reuse the same ID to add trials later; `--count` is the number of additional
trials for that agent invocation. No sweep-wide run cap is configured.
Changing a local YAML file does not change an already registered sweep; register
a new sweep to use a different search space.

The maximum training budget is 1,000 epochs. Periodic rollouts run every 2,000
training steps. Plateau patience starts at the first rollout at or after 100
MSE epochs or 300 flow epochs; ten subsequent evaluations without a reward gain
of at least 0.005 trigger stopping. Small gains accumulate relative to the last
meaningful improvement. Checkpoint selection independently retains the highest
observed reward, with earlier steps winning ties.

On plateau or the epoch limit, evaluate the best checkpoint over 1,000 episodes
and report `final/mean_reward`, the Bayesian objective. Final evaluation uses
separate fixed random streams from checkpoint selection. This remains tuning
data because the search observes the result, not a held-out test estimate.
Weight decay is sampled log-uniformly from positive bounds; zero is excluded.
Videos are disabled; retain the best checkpoint plus latest.

Best-checkpoint final evaluation writes `final_eval.json` inside the run directory
and logs `final/mean_reward`, `final/num_episodes`, `final/checkpoint_step`, and
`final/selection_mean_reward` to W&B. The stopping reason is `plateau` or
`max_epochs`. Periodic scores and checkpoint ranks remain unchanged by final
evaluation. Final evaluation records no videos and uses dedicated final
environment/policy stream identities, with the configured evaluation variations.

Ordinary CLI training remains supported. Early stopping is disabled by default
(`early_stopping_patience=None`), and best-checkpoint final evaluation is opt-in.
It requires checkpoint saving and `checkpoint_top_k >= 1`. Without best-checkpoint
mode, the existing last-step final-evaluation behavior remains unchanged when
training reaches its epoch limit. For plateau-stopped runs, enable best-checkpoint
mode to obtain the separate final evaluation and sweep objective.

The learning-rate/weight-decay bounds, warm-up durations, and stopping threshold
are initial choices to revisit after inspecting the first batch of trials.
