# Inspect a run's results

Use this sequence after a training run completes or fails. You need its experiment directory and the invocation used to create it. Work from **`hw1/`**; the default directory is `exp/seed_<seed>_<timestamp>[_suffix]/`.

## 1. Establish whether the run completed

Check the terminal output and process status first. A normal run performs final evaluation before the interactive progress display ends at `Done` (or `Stopped (plateau)` for early stopping).

- A traceback or nonzero exit means the run failed, even if CSV rows or checkpoints exist.
- Non-interactive output may suppress progress bars.
- A final CSV row records completed evaluation, but subsequent W&B reporting or cleanup can still fail.
- Read cleanup warnings before assuming disk contents match the retention limit.

If CSV logging was disabled, there may be no metric files. The [output controls](../test-runs.md#limit-external-side-effects) explain which files to expect.

## 2. Read metrics in diagnostic order

| Order | File | Inspect |
| --- | --- | --- |
| 1 | `final_eval.csv` | Final `mean_reward`, `num_episodes`, `checkpoint_step`, `selection_mean_reward`, and `stop_reason`. |
| 2 | `eval.csv` | Selection-score history and `video_paths`; locate the selected step. |
| 3 | `validation.csv` | Held-out `loss_mean` and `examples_count`. |
| 4 | `train.csv` | `loss_window_mean`, `window_examples_count`, and progress through optimizer steps. |

Each file uses `global_step`. Training `epoch` is zero-based; a reporting window can cross epochs. Do not average rows blindly when their example counts differ.

To inspect existing files without modifying them, replace the example directory below and run:

```bash
uv run --no-sync python - <<'PY'
import csv
from pathlib import Path

run_dir = Path("exp/REPLACE_WITH_RUN_DIRECTORY")
if not run_dir.is_dir():
    raise SystemExit(f"Run directory not found: {run_dir}")
for name in ("final_eval.csv", "eval.csv", "validation.csv", "train.csv"):
    path = run_dir / name
    if not path.exists():
        print(f"{name}: absent")
        continue
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    print(f"{name}: {len(rows)} rows; last={rows[-1] if rows else None}")
PY
```

Expected output is a row count and last row for each present CSV, or an explicit absence message. This reads the entire CSV into memory and is intended for ordinary local run logs. It does not compute a new evaluation score.

## 3. Locate the policy that was assessed

With persistent checkpoints enabled, use the `checkpoint_step` from `final_eval.csv` to find:

```text
exp/<run>/checkpoints/policy_step_<checkpoint_step>.pt
```

The highest numbered checkpoint is latest, not necessarily best. Default retention keeps the best three plus latest; ties favor later steps. Cleanup failures can leave additional files.

With `--no-save-checkpoints`, successful runs remove their temporary checkpoint directory. A metric row can therefore legitimately refer to a checkpoint that no longer exists. Failed runs may retain temporary checkpoints; the temporary prefix contains the run name.

For an available policy file, follow [Save, load, and train further](save-load-and-continue.md). See [retention behavior](../concepts/checkpoints.md) before treating an untracked or leftover file as a selected checkpoint.

## 4. Use videos to explain behavior

Selection videos live under `videos/step_<step padded to 8 digits>/`. Final-evaluation videos live under `videos/best_step_<step padded to 8 digits>/`. Individual filenames use `episode_<index padded to 7 digits>.mp4`.

- The `video_paths` cell in `eval.csv` is a JSON list of paths relative to the run directory. Use a JSON parser rather than splitting it on commas.
- `final_eval.csv` does not list final video paths; use its selected step and the directory layout.
- Only the first `num_video_episodes` episodes are recorded, bounded by the actual evaluation budget.
- Videos and W&B caches are not removed by policy-checkpoint retention.

Inspect a few rollouts for missed contacts, unstable motion, and whether action chunks carry the pusher past useful states. Treat these as diagnostic observations; the aggregate score still comes from all evaluated episodes.

## 5. Explain surprising results

| Observation | What to check next |
| --- | --- |
| Training loss improves but rollouts do not. | Validation trend, normalization, action semantics, and behavior after the policy leaves demonstration states. |
| Final score differs from selection score. | Separate evaluation streams and budgets; it is not a rerun of identical episodes. |
| A file is absent. | Logging/video flags, temporary storage, failure point, and whether cleanup already ran. |
| Best score seems missing from final assessment. | Retention K, selected step, cleanup warnings, and whether the run completed. |
| Two runs disagree. | Exact settings, dataset, code revision, and variation across training seeds. |

Use [metric interpretation](../concepts/evaluation.md) to avoid reading `mean_reward` as success rate or cumulative return. Before declaring an improvement, follow [fair comparisons](compare-experiments.md).

For dataset, environment, configuration, or loading failures, continue to [Troubleshooting](../troubleshooting.md).
