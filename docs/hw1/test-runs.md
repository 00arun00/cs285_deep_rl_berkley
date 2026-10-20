# Run a small local training test

Exercise training, validation, and policy evaluation without W&B uploads, metric CSVs, or recorded videos. This recipe applies to the Push-T imitation trainer in `hw1`; it runs real training, not a dry run.

## Before you run

1. Complete [environment setup](setup.md) and its CLI-help check.
2. Open a terminal in **`hw1/`**. Allow dataset download if the data is absent, or supply an existing dataset as described below.
3. Keep the complete flag set: each output control disables a different side effect.

The expectations below were checked against the implementation; this recipe was **not executed as part of writing this guide**. No measured runtime or score is implied.

## Run locally

From `hw1/`, run:

```bash title="hw1 · terminal"
uv run src/hw1_imitation/train.py \
  --log-wandb False \
  --log-csv False \
  --save-checkpoints False \
  --num-video-episodes 0 \
  --num-epochs 1 \
  --eval-episodes 1 \
  --final-eval-episodes 1
```

No W&B login is needed. The trainer uses an available accelerator, or CPU if none is available. Runtime depends on the machine; one epoch still processes the training loader; its final incomplete batch is dropped.

## Recognize successful completion

With the recipe unchanged, the source defines this sequence:

1. A **Push-T · Training summary** table shows the effective configuration, dataset, device, and output directory.
2. Training runs for one epoch. Validation and selection rollouts run at their configured intervals and at the final training step.
3. The best retained checkpoint is loaded for a separate one-episode final evaluation.
4. The process returns successfully after cleanup. In an interactive terminal, the progress description ends at **Done**; non-interactive output may suppress progress bars.

A returned shell prompt alone is not proof of success: check that there is no traceback and that the process exits with status zero. In a POSIX shell, run `echo $?` immediately after the training command to inspect its status.

This recipe deliberately leaves no metric CSVs, videos, or persistent policy checkpoints. The experiment directory remains, while the temporary checkpoint directory is removed after successful completion. Final scores are not persisted with both CSV and W&B logging disabled.

## Limit external side effects

| Flag | Effect |
| --- | --- |
| `--log-wandb False` | Skip W&B initialization, metrics, video uploads, and checkpoint artifacts. |
| `--log-csv False` | Do not create or append training, validation, evaluation, or final-evaluation CSVs. |
| `--save-checkpoints False` | Write checkpoints into a unique temporary directory. Remove that directory on success; retain it on failure. |
| `--num-video-episodes 0` | Disable video recording for selection and final evaluation. |

`--save-checkpoints False` does not disable checkpoint creation or W&B uploads by itself. Pair it with `--log-wandb False` for this local test. Temporary checkpoints are needed to load the best retained policy for final evaluation.

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

Use the local command above to avoid Modal execution. `WANDB_MODE=offline` is not a substitute for `--log-wandb False`: the trainer rejects offline W&B runs.

## Inspect available options

```bash
uv run src/hw1_imitation/train.py --help
```

The command uses the same `hw1` working directory. Help displays the current flags and defaults without starting training.

## If the run fails

| Symptom | Next check |
| --- | --- |
| Import error or missing script | Confirm the `hw1/` working directory and repeat the [setup verification](setup.md#verify-the-setup). |
| Dataset download or extraction error | Check network access, disk space, and the dataset layout under `--data-dir`. |
| W&B login prompt or offline-mode error | Confirm `--log-wandb False` is present; `WANDB_MODE=offline` alone does not disable W&B. |
| `No training batches` | The configured batch size exceeds the available training examples; reduce `--batch-size` or inspect the dataset. |
| Temporary checkpoints remain | Inspect the exception; failures retain temporary checkpoints and cleanup errors can leave partial contents. |

Keep the traceback when diagnosing a failure. Run the relevant [automated checks](tests-and-checks.md) to separate code regressions from data or environment issues.

For more symptom-specific checks, see [Troubleshooting](troubleshooting.md).
