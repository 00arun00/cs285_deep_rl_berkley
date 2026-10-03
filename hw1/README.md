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

These assignments use [Weights & Biases (WandB)](https://wandb.ai) for experiment tracking. WandB is a tool for logging and visualizing machine learning experiments. It is free for academic use. Before running a training script, you will need to log in to WandB using your API key.

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

## Reusing policy checkpoints

After each rollout evaluation, including the final training step, the driver
saves `policy_step_<step>.pt` under the W&B run's `checkpoints/` directory and
uploads it as a model artifact. Every evaluation checkpoint is retained; there
is currently no overwrite or top-K retention policy.

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
with torch.no_grad():
    normalized_actions = model.sample_actions(
        states,
        num_steps=inference_config["flow_num_steps"],
    )
actions = normalizer.denormalize_action(normalized_actions.cpu().numpy())
# Clip actions to the environment's bounds before stepping it.
```

The loader returns the model in evaluation mode. For rollout evaluation, pass
`model.chunk_size` and the saved `flow_num_steps` to `evaluate_policy()`. You can
explicitly choose a different flow integration step count when evaluating.
