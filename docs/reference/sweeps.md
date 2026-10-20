# Sweep configuration

Use this reference when reading or changing the W&B definitions in
`hw1/conf/sweeps/`. For registration and execution, follow
[Run a hyperparameter sweep](../hw1/guides/run-sweep.md).

These are W&B configuration files consumed by `wandb sweep`. The `conf/` directory
organizes configuration; it does not enable Hydra composition.

## Choose the source definition

The checked-in files are the source of exact search ranges and budgets:

- [MSE sweep YAML](https://github.com/00arun00/cs285_deep_rl_berkley/blob/main/hw1/conf/sweeps/mse.yaml).
- [Flow sweep YAML](https://github.com/00arun00/cs285_deep_rl_berkley/blob/main/hw1/conf/sweeps/flow.yaml).

Those links show `main`; when working on a branch or older release, inspect the
files in that checkout instead.

Both search hidden widths, chunk size, action-chunk padding, learning rate, and
weight decay. Flow additionally searches integration steps. Each definition fixes
its policy family and uses Bayesian search.

## Read parameter specifications

| YAML form | Meaning | Example |
| --- | --- | --- |
| `value` | Fixed for every trial. | `policy_type: {value: mse}` |
| `values` | Choose from a categorical set. | `chunk_size: {values: [4, 8, 16]}` |
| `distribution`, `min`, `max` | Sample within the given bounds. | Learning rate and weight decay use `log_uniform_values`. |

For `log_uniform_values`, the bounds are actual positive parameter values;
sampling is uniform in their logarithms. See the
[W&B configuration reference](https://docs.wandb.ai/models/sweeps/sweep-config-keys)
for supported distributions and configuration keys.

Tuple parameters are quoted Python tuple strings, such as `"(128, 128, 128)"`.
Boolean choices are YAML booleans, such as `[true, false]`. W&B expands selected
values into arguments accepted by the [shared CLI](training.md#command-line-syntax).

## Follow the training command

```yaml title="Command used by both definitions"
command:
  - ${env}
  - uv
  - run
  - python
  - -m
  - hw1_imitation.train
  - ${args}
```

`${args}` supplies sampled parameters as `--name=value` arguments. The agent runs
the command from its working directory, so launch it from `hw1/`. No separate
sweep trainer is involved. Keep the command paired with the explicit-value parser;
old bare-flag or space-separated tuple syntax is incompatible.

## Distinguish sweep settings from training defaults

Parameters omitted from the YAML inherit [TrainConfig defaults](training.md).
In particular, the supplied definitions leave the seed, selection episode count,
logging controls, and checkpoint storage mode at those defaults. Fixed seeds make
these searches comparisons within one random-stream family, not multi-seed studies.

Both files explicitly use `early_stopping_min_delta: 0.001`, while the ordinary
training default remains `0.005`. Other fixed fields set the epoch cap, evaluation
intervals, final episode budget, warm-up, patience, checkpoint retention, and video
count. Read their values from the YAML before registering a sweep.

Patience counts stale selection evaluations after warm-up, not epochs. Improvement
must strictly exceed the threshold. Changing the evaluation interval therefore
changes the amount of training between stopping decisions. See
[early stopping](../hw1/concepts/training-lifecycle.md#early-stopping-and-final-evaluation).

The sweep's `project` names its W&B destination. If changing it, keep the trainer's
`wandb_project` setting consistent as well. Budget and configuration changes belong
in a newly registered definition; preserve the original definition and code revision
when comparing searches.

## Objective and final assessment

Both definitions specify `eval/mean_reward` with `goal: maximize`. The logger
configures a maximum summary for that metric, preserving the best periodic score
for inspection. `final/mean_reward` is reported after evaluating the selected
checkpoint on separate streams; it is not named as the sweep objective.

See [metric interpretation](../hw1/concepts/evaluation.md#sweep-objective-and-final-assessment)
for the distinction between search feedback and final assessment. The sweep does
not change checkpoint storage or final-evaluation behavior.
