# Change the checkpoint format

Use this guide when a policy change requires new serialized state or different interpretation of existing fields. Start by reading [the current checkpoint contract](../concepts/checkpoints.md) and identifying which consumers would be affected.

This page describes how to plan and verify a future implementation. The current format remains **version 1**, and the current loader rejects unsupported versions. There is no automatic migration path implemented here.

Source paths below are relative to **`hw1/src/hw1_imitation/`**. Run existing checks from **`hw1/`** with the [development environment](../setup.md).

## 1. Define the contract change

Record the old and proposed meaning, not just a new dictionary key:

| Question | Why it matters |
| --- | --- |
| Is this architecture, learned state, normalization, runtime training configuration, or inference configuration? | Determines reconstruction and which component owns the field. |
| Is a missing value safe to default? | A silent default can change behavior without an obvious load failure. |
| Can old weights and normalizers still represent the model? | Shape compatibility alone does not establish semantic compatibility. |
| Does the proposal add optimizer/RNG/training position? | That changes the policy-only contract and needs an explicit resumption design. |
| Which files must remain loadable? | Establishes the compatibility target before implementation. |

Trace `save_policy`, `load_policy`, `PolicyConfig`, policy construction, dropout restoration, and the trainer's `--init-from` path. Check direct inference users as well as training.

## 2. Choose a version and compatibility policy

For an incompatible interpretation or required new content, plan a version bump rather than silently reinterpreting version 1.

| Proposed policy | Implementation responsibility |
| --- | --- |
| Reject older versions explicitly. | Keep a clear unsupported-version error and document the boundary. |
| Support selected older versions. | Add an explicit conversion/reader path, validate converted state, and define supported source versions. |
| Export a separate conversion tool. | Preserve original files and specify the new output format and verification. |

These are design alternatives, not current loader capabilities. Agree on one before implementing it. Do not claim compatibility solely because a version field was changed, and do not weaken `weights_only=True` to accept legacy whole-model pickle files.

## 3. Update writer and reader together

1. Change the schema constant and payload assembly in `checkpoint.py` according to the chosen policy.
2. Make the loader validate the new value's type, bounds, dimensions, and relationships before applying it where possible.
3. Reconstruct architecture and strictly load weights; preserve matching dropout-layer requirements or explicitly migrate their names.
4. Preserve CPU serialization, normalization copies, inference defaults, requested-device placement, and evaluation mode unless a deliberate contract change says otherwise.
5. Trace the trainer: loaded architecture and normalization currently win over fresh-model flags, while new-run optimization/evaluation settings come from the invocation.

Keep the atomic write sequence: serialize to a sibling temporary file, replace the destination after success, and clean up. It still does not support concurrent writers to the same destination or guarantee crash durability.

## 4. Review the compatibility matrix

Use the matrix as acceptance criteria for the implementation task. Existing tests cover several current-format cases; they do not automatically prove a proposed format works. Update relevant tests, and agree on any additional coverage required by that task.

| Case | Required outcome |
| --- | --- |
| New writer → new reader | Equivalent architecture, normalized/raw predictions, and inference settings. |
| Existing version 1 file → new reader | Explicitly agreed migration or clear rejection. |
| New file → old reader | Unsupported-version rejection when the version changes. |
| Missing/malformed required field | Actionable failure, without partially publishing a checkpoint. |
| Incompatible weights/statistics/dropout names | Rejection unless a deliberate, validated conversion handles it. |
| Failed serialization over an existing file | Preserve the old destination. |
| Loaded model used for inference | Correct device, evaluation mode, and caller-owned RNG. |
| Loaded model used for further training | Fresh optimizer/counters/RNG, unless a separately designed contract changes this. |

Compare raw-space predictions as well as weights: an altered normalizer can change behavior with identical parameters. For stochastic policies, use matching generator identities and sampling settings during comparison.

## 5. Run the existing checks

```bash
uv run --no-sync pytest \
  tests/test_checkpoint.py tests/test_checkpoint_atomicity.py \
  tests/test_checkpoint_retention.py tests/test_checkpoint_retention_properties.py \
  tests/test_dropout_integration.py -v
```

Expect current serialization, restoration, atomicity, and retention behavior to remain valid or to be intentionally updated for the agreed contract. Then run the [full CI sequence](../tests-and-checks.md).

Finally, update the schema documentation, load examples, and compatibility statement alongside the implementation. Preserve fixture provenance; generating both “old” and “new” files through the new writer does not test backward compatibility.

## Common mistakes

- Adding a required field without deciding how existing files fail or migrate.
- Calling an initialization-from-weights workflow exact training resumption.
- Assuming serialization success proves behavioral equivalence.
- Renaming modules without considering weight keys and serialized dropout names.
- Treating retention metadata or W&B aliases as the policy schema itself.

For today's behavior and commands, use [Save, load, and train further](save-load-and-continue.md).
