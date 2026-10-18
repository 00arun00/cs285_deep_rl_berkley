# Run a small local training test

Exercise training, validation, and policy evaluation without W&B uploads, metric CSVs, or recorded videos. This recipe applies to the Push-T imitation trainer in `hw1`; it runs real training, not a dry run.

## Run locally

With `uv` installed, open a terminal in `hw1` and run:

```bash title="hw1 · terminal"
uv run src/hw1_imitation/train.py \
  --no-log-wandb \
  --no-log-csv \
  --no-save-checkpoints \
  --num-video-episodes 0 \
  --num-epochs 1 \
  --eval-episodes 1 \
  --final-eval-episodes 1
```

No W&B login is needed. The trainer uses an available accelerator, or CPU if none is available. Runtime depends on the machine; one epoch still processes the training split.

## Limit external side effects

| Flag | Effect |
| --- | --- |
| `--no-log-wandb` | Skip W&B initialization, metrics, video uploads, and checkpoint artifacts. |
| `--no-log-csv` | Do not create or append training, validation, evaluation, or final-evaluation CSVs. |
| `--no-save-checkpoints` | Write checkpoints into a unique temporary directory. Remove that directory on success; retain it on failure. |
| `--num-video-episodes 0` | Disable video recording for selection and final evaluation. |

`--no-save-checkpoints` does not disable checkpoint creation or W&B uploads by itself. Pair it with `--no-log-wandb` for this local test. Temporary checkpoints are needed to load the best retained policy for final evaluation.

## Reduce the run size

| Flag | Effect |
| --- | --- |
| `--num-epochs 1` | Train for one epoch instead of the default 400. |
| `--eval-episodes 1` | Use one episode per selection evaluation instead of 100. |
| `--final-eval-episodes 1` | Evaluate the best retained checkpoint for one episode instead of 100. |

Validation and selection evaluation remain enabled, including selection evaluation at the last training step. Final evaluation runs separately. These small episode budgets check that the pipeline runs; they are insufficient for a reliable policy-quality comparison.

## What still gets written

- **Experiment directory:** `exp/<run-name>/` is still created under `hw1`, even with all four side-effect controls set.
- **Temporary checkpoints:** files are written to the system temporary directory. Failed runs retain them for inspection, subject to operating-system cleanup.
- **Dataset:** missing data may be downloaded and extracted under `data/`.
- **Python environment:** `uv run` may install dependencies and update its environment/cache when needed.

To reuse an existing dataset, append `--data-dir /path/to/data-root`. That directory should contain `pusht/pusht_cchi_v7_replay.zarr`. An existing `pusht.zip` can also be extracted without downloading it again. These flags do not enforce network isolation.

Use the local command above to avoid Modal execution. `WANDB_MODE=offline` is not a substitute for `--no-log-wandb`: the trainer rejects offline W&B runs.

## Inspect available options

```bash
uv run src/hw1_imitation/train.py --help
```

The command uses the same `hw1` working directory. Help displays the current flags and defaults without starting training.
