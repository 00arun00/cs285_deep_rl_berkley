# Run a hyperparameter sweep

Use a W&B sweep to search training configurations for an MSE or flow policy.
Each trial runs the same trainer as an individual command. W&B chooses parameters;
an agent on your machine launches the trials.

## Before you run

1. Complete [setup](../setup.md) and a [small local training test](../test-runs.md).
2. Authenticate using the [W&B setup instructions](../setup.md#enable-wb-reporting-only-when-needed).
3. Work from **`hw1/`** so configuration paths, data, and experiment outputs resolve correctly.

The supplied configurations launch real training and upload metrics and checkpoint
artifacts. They use substantial training and evaluation budgets; they are not
smoke tests. The YAML-to-CLI path is covered by automated tests with training
mocked. No live sweep was run to validate this guide, and no runtime or score is promised.

## 1. Choose and inspect a configuration

| File, relative to `hw1/` | Policy | Additional search |
| --- | --- | --- |
| `conf/sweeps/mse.yaml` | Direct action-chunk prediction | Shared architecture and optimization parameters. |
| `conf/sweeps/flow.yaml` | Flow-based action-chunk prediction | Also searches flow integration steps. |

Read the [configuration reference](../../reference/sweeps.md) before changing a
search space or budget. Both configurations maximize `eval/mean_reward`, whose
W&B summary is configured to retain its maximum. Final evaluation uses separate
random streams and is not the configured search objective.

## 2. Register the sweep

Run the command for the policy you chose:

```bash title="hw1 · register an MSE sweep"
uv run --no-sync wandb sweep conf/sweeps/mse.yaml
```

```bash title="hw1 · register a flow sweep"
uv run --no-sync wandb sweep conf/sweeps/flow.yaml
```

To register a new sweep with a different display name without editing the YAML,
pass `--name`. This overrides the YAML's `name` for the new sweep:

```bash title="hw1 · register a flow sweep with a custom name"
uv run --no-sync wandb sweep --name "flow-experiment-2" conf/sweeps/flow.yaml
```

The same option works with `conf/sweeps/mse.yaml`.

Registration contacts W&B and returns a sweep ID and URL; it does not launch
training. Save the full `ENTITY/PROJECT/SWEEP_ID` printed by W&B. Registering both
files creates separate sweeps with separate IDs. See the
[W&B registration documentation](https://docs.wandb.ai/models/sweeps/initialize-sweeps).

## 3. Launch a bounded number of trials

Replace the placeholder below with the full ID from registration:

```bash title="hw1 · run up to five trials"
uv run --no-sync python -m hw1_imitation.sweep_agent --count 5 ENTITY/PROJECT/SWEEP_ID
```

This starts real training on the machine running the agent. The count limits this
agent invocation, not the entire sweep; the YAMLs do not set a sweep-wide run cap.
Run one agent at a time initially to avoid competing for local compute.

The [launcher][hw1_imitation.sweep_agent.main] accepts a positional `sweep_id`
and optional `--count`; omitting the count leaves the agent without a local trial
limit. It suppresses DEBUG and INFO messages from the parent W&B agent, including
`Running runs:`, so routine status messages do not interrupt the training progress
bar. Agent warnings and errors remain visible and can still interrupt the bar.
The training subprocess's W&B startup messages, links, run history, and summary
remain visible.

This launcher filters W&B's internal `wandb.wandb_agent` logger; it does not set
`WANDB_QUIET` or `WANDB_SILENT`. Recheck that logger integration when upgrading W&B.
The sweep YAML still launches `hw1_imitation.train` for each trial.

Each trial creates its own experiment directory under `exp/`, logs to W&B, and
retains checkpoints. The supplied YAMLs disable videos but leave CSV logging and
persistent checkpoint storage enabled through training defaults. See
[checkpoint behavior](../concepts/checkpoints.md) for retention and failure handling.

## 4. Continue or change the search

To request more trials for the same registered definition, reuse its ID:

```bash title="hw1 · request five more trials"
uv run --no-sync python -m hw1_imitation.sweep_agent --count 5 ENTITY/PROJECT/SWEEP_ID
```

Keep the code revision and environment recorded with the sweep. Reusing the ID
does not restore your local code to its earlier state. If you change the local
YAML, register a new sweep to use that definition; editing a file does not update
the previously registered definition.

## 5. Verify results before comparing trials

1. Open the returned sweep URL and check trial status and sampled configuration.
2. Inspect the maximum periodic evaluation reward alongside the full reward history.
3. Confirm that a completed trial reports `final/mean_reward` and its selected checkpoint.
4. Follow [Inspect results](inspect-results.md#inspect-sweep-trials) for local artifacts
   and [Compare experiments](compare-experiments.md) before declaring an improvement.

An objective value can exist before a trial completes. A failed or interrupted
trial is not a completed final assessment. Confirm promising settings across
multiple seeds instead of reporting only the strongest search result.

## If the sweep fails

| Symptom | Next check |
| --- | --- |
| YAML file not found | Run from `hw1/` and use `conf/sweeps/`, not the former `sweeps/` path. |
| Authentication or project error | Check W&B login, access to the project, and the full sweep ID. |
| Trial rejects an argument | Check the [explicit-value CLI syntax](../../reference/training.md#command-line-syntax) and that the agent uses this checkout. |
| Local YAML edits do not affect trials | Register the changed definition and use its new ID. |
| Final metrics are absent | Inspect the trial process status and traceback; training or final evaluation may not have completed. |

Continue to [Troubleshooting](../troubleshooting.md) for dataset, device, and checkpoint failures.
