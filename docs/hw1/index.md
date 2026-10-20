# Start here: Push-T imitation learning

The implementation in `hw1/` trains a policy to imitate recorded demonstrations of **Push-T**: moving a T-shaped block into a target pose with a controllable pusher. A **policy** maps the current observation to actions; this implementation uses numeric state observations rather than camera images.

These guides assume you know Python, PyTorch, and basic supervised learning. Use this page to run or inspect the Push-T implementation. For other projects and their status, return to the [repository overview](../index.md).

## Get started

1. **Prepare your environment:** follow the [HW1 setup instructions](setup.md). Use the `hw1/` working directory for HW1 commands.
2. **Exercise the pipeline:** follow [Run a small local training test](test-runs.md). It explains output controls and remaining side effects. This is real training; it can download data and install dependencies.
3. **Check a code change:** use the [tests and checks](tests-and-checks.md) for lint, formatting, and pytest commands.
4. **Understand the architecture:** follow the [conceptual reading path](#understand-the-system) below, then use the source map and [Python reference](../reference/index.md) for implementation details.

For an existing trained policy, start with [checkpoint loading and further training](guides/save-load-and-continue.md). Further training starts a fresh optimizer and run; it does not resume an interrupted training trajectory.

## What is supported

Training learns from recorded state–action pairs. Both supported policies predict an **action chunk**: a sequence of actions from one observation. `mse` predicts the chunk directly; `flow` transforms sampled noise into a chunk using a learned velocity field.

Environment rollouts assess the learned behavior and select checkpoints. Validation loss measures demonstration fit; rollout scores measure behavior under the policy's own actions. Their definitions and interpretation belong in the [evaluation guide](concepts/evaluation.md).

## Understand the system

Read the first four pages in order for the path from demonstrations to results. Use the final two when you need reproducibility or saved-policy details.

| Page | Question it answers |
| --- | --- |
| [Push-T task and dataset](concepts/pusht.md) | What are the observations, actions, demonstrations, and supported policy families? |
| [Demonstrations to training batches](concepts/data-pipeline.md) | How do episode splits, normalization, chunks, and padding create examples? |
| [Training lifecycle](concepts/training-lifecycle.md) | How do optimization, monitoring, checkpoint selection, and final evaluation fit together? |
| [Evaluation and results](concepts/evaluation.md) | What do the metrics mean, and how should comparisons account for variability? |
| [Randomness and reproducibility](concepts/randomness.md) | Which identities and generator lifetimes control random draws? |
| [Checkpoint format and lifecycle](concepts/checkpoints.md) | What survives saving, loading, retention, and cleanup? |

## Read the implementation

Implementation modules below are relative to `hw1/src/hw1_imitation/`; tests are in `hw1/tests/`.

| Entry point or module | Start here when you want to understand… |
| --- | --- |
| `train.py` | CLI configuration and the training lifecycle. |
| `data.py` | Demonstration episodes, normalization, and action-chunk datasets. |
| `model.py` | MSE and flow policies, their losses, and action sampling. |
| `evaluation.py` | Validation, environment rollouts, and checkpoint selection/retention. |
| `checkpoint.py` | Saving and reconstructing a policy and its normalizer. |
| `randomness.py`, `rng_modules.py` | Named random streams and layers that consume explicit generators. |
| `logging_utils.py`, `early_stopping.py` | Run outputs and stopping when rollout improvement stalls. |
| `modal_train.py` | The optional remote-training entry point. |

For a first code reading, follow `train.py` → `data.py` → `model.py` → `evaluation.py`. Return to the supporting modules when you need their specific behavior.

## Common tasks

- [Configure and compare experiments](guides/compare-experiments.md).
- [Run a hyperparameter sweep](guides/run-sweep.md).
- [Inspect results and diagnose a run](guides/inspect-results.md).
- [Save, load, and train further](guides/save-load-and-continue.md).
- [Add a random stream](guides/add-random-stream.md).
- [Change the checkpoint format](guides/change-checkpoint-format.md).
- [Change or add a policy](guides/change-policy.md).

For failures or unexpected behavior, use [Troubleshooting](troubleshooting.md) to choose a diagnostic path.

Individual commands and sweep trials use the same training pipeline. Use a direct
command for a chosen configuration, or a W&B sweep to search across configurations
defined in `hw1/conf/sweeps/`. See [CLI syntax](../reference/training.md#command-line-syntax)
and [sweep configuration](../reference/sweeps.md) for their input formats.
