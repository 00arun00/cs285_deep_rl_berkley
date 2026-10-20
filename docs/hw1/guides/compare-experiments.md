# Compare experiments fairly

Use this guide when deciding whether a policy or training-setting change improves Push-T behavior. First complete [setup](../setup.md) and a [small local run](../test-runs.md). Run commands from **`hw1/`**.

The commands below launch real training, may download demonstrations, and create outputs. They were checked against the CLI and source, not executed to produce benchmark results for this guide.

## 1. Define the comparison before launching

Write down the changed factor and the settings held constant:

| Record | Why it matters |
| --- | --- |
| Code revision, environment, dataset path/content | Different inputs or implementations can change results. |
| Policy, hidden widths, chunk size, dropout | Defines the prediction problem and model. |
| Epochs, batch size, learning rate, weight decay | Defines optimization and update count. |
| Root seed and stream variations | Controls split, initialization, training noise, and evaluation. |
| Selection/final episode budgets and intervals | Controls score variability and selection opportunities. |
| Retention and early stopping | Determines which checkpoint reaches final evaluation and how long training runs. |
| Flow integration steps | Changes flow inference cost and potentially quality. |

Keep matching seeds across alternatives, then repeat the comparison across several root seeds. Report the spread as well as the average. Matching seed numbers does not give different architectures identical weights or identical random draw consumption.

## 2. Name runs and retain evidence

For example, compare MSE with dropout disabled versus probability 0.2:

```bash
uv run --no-sync src/hw1_imitation/train.py \
  --policy-type mse --dropout-p 0 --seed 42 \
  --num-epochs 20 --batch-size 128 --lr 0.0003 \
  --eval-episodes 5 --final-eval-episodes 100 \
  --checkpoint-top-k 3 --num-video-episodes 0 --log-wandb False \
  --exp-name mse-dropout0

uv run --no-sync src/hw1_imitation/train.py \
  --policy-type mse --dropout-p 0.2 --seed 42 \
  --num-epochs 20 --batch-size 128 --lr 0.0003 \
  --eval-episodes 5 --final-eval-episodes 100 \
  --checkpoint-top-k 3 --num-video-episodes 0 --log-wandb False \
  --exp-name mse-dropout02
```

These are illustrative budgets, not tuned recommendations. Both retain the current defaults for unspecified settings, including evaluation intervals and disabled early stopping. Repeat both with the same additional seeds, such as 43 and 44.

`--exp-name` appends a suffix; it does not replace the complete directory name. Expect separate `exp/seed_<seed>_<timestamp>_<suffix>/` directories, metric CSVs, and retained checkpoints. W&B and videos are disabled here; CSV logging and persistent checkpoints remain enabled.

Save the exact invocation and code revision with your experiment notes. Local CSVs and policy checkpoints do not capture the complete invocation. W&B records configuration when enabled, but a local-only run should not depend on that service for its records.

## 3. Change policy or randomness deliberately

- **MSE versus flow:** change `--policy-type flow` and explicitly record `--flow-num-steps`. Keep the dataset/split and evaluation settings paired; compare rollout scores, not their different loss objectives.
- **A different split:** change `--data-split-variation` while keeping the root seed fixed. This changes train/validation episode membership.
- **A different rollout setup:** change `--eval-env-variation` or `--eval-policy-variation` independently. Final counterparts are `--final-eval-env-variation` and `--final-eval-policy-variation`.
- **Training variability:** vary the root seed across repeated runs; changing a root seed affects all derived identities, not only initialization.

See [random-stream identities](../concepts/randomness.md) before selecting variations. MSE inference in evaluation mode does not consume policy-sampling noise, so changing that stream alone may not change its result.

For a plateau-stopping experiment, set an explicit policy such as `--early-stopping-patience 5 --early-stopping-min-epochs 10 --early-stopping-min-delta 0.005` on each compared run. These are illustrative values. Patience counts stale selection evaluations after warm-up, so also record `--eval-interval`; equal epoch budgets no longer imply equal work. See the [training lifecycle](../concepts/training-lifecycle.md#early-stopping-and-final-evaluation) for the strict improvement and warm-up rules.

Dropout accepts `0 <= p < 1`; explicit probabilities apply to hidden activations. With fresh models, omission means zero. With loaded models, omission inherits saved per-layer probabilities; see [checkpoint initialization](save-load-and-continue.md).

## 4. Compare the selected policies

1. Inspect `final_eval.csv` for the final score, episode count, selected `checkpoint_step`, and selection score.
2. Trace that step in `eval.csv`; inspect training and validation histories for context.
3. Check that budgets, stream settings, and retention policies match your comparison plan.
4. Aggregate across seeds, preserving individual results and failures rather than reporting only the best run.

Smaller selection budgets make checkpoint choice noisier even when the final budget is large. K=0 normally evaluates latest rather than the historical best. Repeatedly changing settings based on final scores also makes those scores part of your selection process; describe that honestly.

## Verify and avoid misleading conclusions

| Check | Mistake it prevents |
| --- | --- |
| Every run completed final evaluation. | Comparing a partial run with a completed run. |
| Compare `mean_reward` under matching scoring rules. | Treating it as cumulative return or success rate. |
| Match update counts when batch size or padding changes. | Assuming equal epochs always mean equal optimization work. |
| Inspect warnings and selected checkpoint identity. | Assuming retention cleanup or checkpoint selection always succeeded. |
| Report multiple seeds and their variability. | Interpreting a single favorable rollout as a robust improvement. |

Continue with [Inspect results](inspect-results.md) and [metric definitions](../concepts/evaluation.md).
