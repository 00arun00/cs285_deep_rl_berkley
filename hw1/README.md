# Homework 1: Imitation Learning

## Setup

Follow [Development setup and tools](../docs/hw1/setup.md) for Python, uv, the
HW1 environment, and required versus optional tools. Run HW1 commands from this
directory.

Then use [Run a small local training test](../docs/hw1/test-runs.md) to exercise
the pipeline with reduced output and evaluation budgets.

## Weights & Biases (wandb) login

See [optional services and tools](../docs/hw1/setup.md) for W&B setup.
Local-only runs with `--no-log-wandb` do not require W&B login.

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

See [Configure and compare experiments](../docs/hw1/guides/compare-experiments.md)
for a paired dropout comparison, and [Save, load, and train further](../docs/hw1/guides/save-load-and-continue.md)
for inherited probabilities and explicit overrides. The
[randomness explanation](../docs/hw1/concepts/randomness.md) covers generator use
and evaluation mode.

## Evaluation episode counts

[Configure and compare experiments](../docs/hw1/guides/compare-experiments.md)
explains selection and final budgets. See [Evaluation and results](../docs/hw1/concepts/evaluation.md)
for metric definitions and [Inspect results](../docs/hw1/guides/inspect-results.md)
for CSVs, checkpoints, and videos.

## Reusing policy checkpoints

[Save, load, and train further](../docs/hw1/guides/save-load-and-continue.md)
covers the procedures. [Checkpoint format and lifecycle](../docs/hw1/concepts/checkpoints.md)
explains saved state, retention, output flags, and failure handling.

### Load for inference or evaluation

Follow the [inference and evaluation examples](../docs/hw1/guides/save-load-and-continue.md).

### Train further from saved weights

Follow the [new-training-run procedure](../docs/hw1/guides/save-load-and-continue.md).
Policy checkpoints do not restore optimizer state, RNG position, or training position.

### Checks

[Tests and checks](../docs/hw1/tests-and-checks.md) covers the full suite,
individual tests, lint and formatting, the contracts under test, and CI commands.

## Early stopping

Stop when rollout reward plateaus. Disabled by default.

| Setting | Default | Purpose |
| --- | --- | --- |
| `early_stopping_patience` | `None` | Enable with a positive number of stale evaluations |
| `early_stopping_min_epochs` | `100` | Warm-up before counting stale evaluations |
| `early_stopping_min_delta` | `0.005` | Reward gain must exceed this to reset patience |
