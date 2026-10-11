# Homework 1: Imitation Learning

## Setup

Use `uv` for dependency management and run Python through `uv run`:

```bash
uv sync --locked --dev
uv run wandb login
```

W&B is enabled by default. Use `logging=local` for local-only runs without W&B credentials.

## Configuration and local training

Run commands from `hw1/`. Hydra is the configuration frontend:

```bash
uv run -m hw1_imitation.cli
uv run -m hw1_imitation.cli experiment=mse_baseline train.optimization.lr=0.0001
uv run -m hw1_imitation.cli model=flow architecture=mlp512 logging=local
uv run -m hw1_imitation.cli 'train.model.hidden_dims=[128,128]' train.evaluation.episodes=5
uv run -m hw1_imitation.cli --cfg job --resolve
```

The last command inspects configuration without importing the trainer or starting training.
Tyro flags and the old flat configuration paths are intentionally removed.

`ExperimentConfig` has `train` and `run` sections. The frozen, validated `TrainConfig`
contains `data`, `model`, `optimization`, `evaluation`, `randomness`, `logging`, and
`checkpoints`. Functions receive concrete Python dataclasses. Hydra objects stay at
composition and CLI boundaries. `ModelConfig` describes requested architecture;
`PolicyConfig` describes the actual model, including data-derived input/output dimensions.

Python field definitions are the source of types, validation, and defaults. Composition
order is Python defaults, selected components, experiment values, then CLI value
overrides. CLI component selections override experiment defaults. Unselected component
slots inherit Python defaults. Unknown fields and invalid values fail before training.
Relative paths remain relative to the invocation directory. Hydra does not change cwd,
configure logging, or create its own output/snapshot directories.

```python
from hw1_imitation.config_composition import compose_experiment

config = compose_experiment(["model=flow", "train.optimization.lr=0.0001"])
```

The `run` section controls `output_root` (default `exp/`) and an optional `name`.
An explicit name selects exactly that directory and must be unused; unnamed runs
get unique names. New run directories never overwrite existing ones.

`launch_experiment(config, overrides=...)` owns run setup and calls
`run_training(config.train, output_dir=...)`. The core trainer requires an existing
directory; it does not select output paths. Both local and Modal entrypoints use
the same setup function.

Each run saves `config.json` (requested, resolved configuration),
`overrides.json` (ordered launch overrides), `manifest.json` (invocation directory,
start time, and parent checkpoint identity), and `run_facts.json` (actual model,
device, dataset path, and split sizes). These files are written even when metric
logging is disabled. They document the run, not a complete environment or dataset
archive. Setup files remain available when training fails.

## Modal

```bash
uv run modal run -m hw1_imitation.modal_train
uv run modal run -m hw1_imitation.modal_train model=flow train.optimization.lr=0.0001
uv run modal run -m hw1_imitation.modal_train experiment=mse_baseline logging=local
```

Modal uses the same composition and trainer. Its `runtime=modal` preset defaults to
`/vol/data` and `/vol/exp` on `hw1-imitation-volume`; user value overrides win. The
volume is committed after successful training. Container resources remain defined
in the thin Modal adapter. No separate Hydra/Tyro Modal apps are needed.

```bash
uv run -m hw1_imitation.cli runtime=modal --cfg job --resolve
uv run modal volume ls hw1-imitation-volume exp
uv run modal volume get hw1-imitation-volume exp/<run-directory>
```

## Evaluation and outputs

The score is the mean of each rollout episode's maximum reward. Periodic and final
budgets default to 100 episodes each. If an interval lands on the final step, only
one final evaluation runs. Final evaluation is not a separate untouched holdout.
Action chunk size controls both prediction horizon and actions executed before
observing again. These training/evaluation semantics are unchanged.

CSV logging, W&B, and checkpoint saving can be disabled independently:

```bash
uv run -m hw1_imitation.cli train.logging.csv=false train.logging.wandb=false train.checkpoints.enabled=false
uv run -m hw1_imitation.cli train.evaluation.num_video_episodes=0
```

When enabled, every evaluation saves a local policy checkpoint and uploads it to
W&B. `train.checkpoints.top_k` (default 3) retains best K plus latest; ties favor
earlier steps, and zero keeps latest only. Local and remote retention agree;
deletion waits for upload completion. Cleanup failures are logged and retried at
later saves, with unresolved steps reported on completion. Videos and W&B cache
files are outside checkpoint retention. Offline W&B is unsupported; use
`logging=local` instead.

## Warm starts and inference checkpoints

```bash
uv run -m hw1_imitation.cli initialization=checkpoint train.warm_start=/path/to/policy.pt train.optimization.num_epochs=20 train.optimization.lr=0.0001
```

This begins a new 20-epoch optimization run from saved weights, with a fresh
optimizer, counters, random streams, and tracking run. The checkpoint supplies
architecture and normalization. `initialization=checkpoint` sets `train.model=null`
so fresh-model settings cannot silently conflict. Direct Python callers use
`TrainConfig(model=None, warm_start=Path(...), ...)`. New demonstrations must have
compatible observation/action dimensions and meanings. Inference settings such
as flow steps come from the new run's evaluation configuration.

Run setup copies the parent checkpoint to `initial_policy.pt`, records its original
absolute path and SHA-256 digest in the manifest, and loads that preserved copy.
The requested config retains the original path. This keeps lineage intact if the
parent run later deletes its checkpoint. The input copy is outside latest + top-K
retention of the new run's output checkpoints.

Policy checkpoints contain architecture, CPU weights, normalization, and inference
defaults. They are written atomically and loaded with `weights_only=True`. They do
not restore optimizer history, RNG state, or data-loading position. Full resume,
curriculum orchestration, and checkpoint-format migrations are outside this stage.

For standalone inference, `load_policy(path, device=...)` returns the model in
evaluation mode, its normalizer, and inference settings. Normalize raw observations,
call `model.sample_actions(..., generator=..., num_steps=...)`, then denormalize actions.

## Checks

GitHub Actions runs `HW1 checks` for PRs targeting any branch and pushes to `main`.

```bash
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check tests
uv run --no-sync pytest -q
```

Tests cover composition, validation, model round trips, warm starts, evaluation
budgets, logging, retention, and random-stream isolation. Tiny training fixtures
use mocked data, environments, and W&B services; no remote training is launched.
