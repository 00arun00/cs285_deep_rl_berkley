# Troubleshoot a Push-T run

Start with the first failing operation, its full traceback, the exact command, and the run directory if one was created. The checks below distinguish configuration, environment, data, and code failures; they are diagnostic steps, not claims that a remedy has been tested on your machine.

Run HW1 commands from **`hw1/`** after [setup](setup.md). Preserve logs and partially written artifacts until you understand the failure.

## Find the relevant symptom

| Symptom | Start here |
| --- | --- |
| Missing command, import, script, or locked dependency | [Environment and working directory](#environment-and-working-directory) |
| Download/extraction error, missing Zarr key, invalid episodes | [Dataset and cache](#dataset-and-cache) |
| Invalid option, no batches, or video dimensions | [Configuration and sample counts](#configuration-and-sample-counts) |
| Device error or out-of-memory failure | [Device and memory](#device-and-memory) |
| Checkpoint will not load | [Checkpoint compatibility](#checkpoint-compatibility) |
| Same seed produces surprising differences | [Randomness and repeatability](#randomness-and-repeatability) |
| Missing metrics, videos, or policy files | [Outputs and cleanup](#outputs-and-cleanup) |
| Local checks and CI disagree | [Local versus CI](#local-versus-ci) |

## Environment and working directory

| Symptom | Check and next step | Verify |
| --- | --- | --- |
| `uv` is not found | Check shell `PATH`; open a new terminal after installation. Follow [setup](setup.md) if needed. | `uv --version` succeeds. |
| Script or `tests/` not found | Confirm the working directory is `hw1/`, not the documentation root. | The relative paths in [tests and checks](tests-and-checks.md) exist. |
| Import error or missing pytest/Hypothesis | Use the HW1 environment and its locked development dependencies. Follow setup's sync procedure. | Trainer `--help` and the named CLI-defaults test in [setup verification](setup.md#verify-the-setup) succeed. |
| Locked synchronization fails | Check that the manifest and lockfile are from the same checkout and that Python meets the project's requirements. Resolve the mismatch rather than bypassing the lock. | Locked synchronization succeeds without an unintended dependency change. |

HW1 selects Python 3.11 through `.python-version`; the root environment is for documentation. A successful site build does not establish that HW1 dependencies are installed.

## Dataset and cache

The expected store is `<data-dir>/pusht/pusht_cchi_v7_replay.zarr`. The loader consumes `data/state`, `data/action`, and `meta/episode_ends`; see [artifact and field contracts](concepts/pusht.md).

| Symptom | Check and supported next step | Verify |
| --- | --- | --- |
| Download fails | Inspect the original network/filesystem error, available disk space, and write access. For a known-good existing dataset, use `--data-dir` to its root. | The intended Zarr store can be opened and its required arrays read. |
| `BadZipFile` or extraction failure | Inspect the cached `pusht.zip` and extraction target. The downloader reuses an existing ZIP and does not repair partial downloads. | A deliberately replaced or separately recovered artifact has the expected layout and readable arrays. |
| Existing directory but missing arrays/keys | The downloader returns immediately when the expected Zarr path exists; existence is not an integrity check. Preserve the partial cache and investigate the missing fields. | All three required fields load, then episode validation succeeds. |
| Unequal packed lengths or invalid episode ends | Compare state/action row counts and exclusive offsets. Do not manufacture offsets to suppress the error. | Ends are positive, strictly increasing, and finish at the total row count. |
| Empty or malformed episodes/statistics | Check data shape, contents, and representation against [the data pipeline](concepts/data-pipeline.md). | Nonempty compatible episodes produce valid finite normalization statistics. |

A failed download can leave an incomplete ZIP; a failed extraction can leave a misleadingly present Zarr directory. Repeating the same command may reuse that state. There is no automatic checksum verification or partial-cache repair.

If recovery requires another download or replacing files, first preserve the failed artifact and decide explicitly which cache to replace. A separate data root avoids overwriting evidence, but can cause a fresh download. No deletion or repair command is prescribed here.

## Configuration and sample counts

These messages originate in the implementation:

| Message or symptom | Next step | Verify |
| --- | --- | --- |
| `<name> must be positive` | Use positive epoch/batch/chunk/interval/episode counts and flow steps. Read current flags with trainer `--help`. | Argument parsing accepts the corrected value. |
| `dropout_p must be in [0, 1)` | Use zero to disable dropout, or a probability below one. | Effective summary reflects the intended dropout settings. |
| `checkpoint_top_k must be nonnegative` | Use K≥0; K=0 means latest-only retention. | Configuration parses; interpret final selection accordingly. |
| `Video dimensions must be positive and even` | Supply positive even dimensions, or set `--num-video-episodes 0` when videos are unnecessary. | Configuration parses; later recording requires the encoding environment too. |
| `No training batches; reduce batch_size or check the dataset.` | Training drops its incomplete final batch. Reduce batch size or inspect available examples. | At least one complete training batch exists. |
| `no valid chunks: reduce chunk_length or enable padding` | Check episode lengths against horizon; the trainer enables final-action padding by default. | Per-episode windows exist without crossing boundaries. |
| `validation loader produced no examples` | Inspect split size and window availability, especially with tiny custom data. | Validation contains examples; do not replace an undefined loss with zero. |

A loaded checkpoint supplies its own chunk horizon, so changing `--chunk-size` does not resize it. See [save/load settings](guides/save-load-and-continue.md#initialize-another-training-run).

## Device and memory

The trainer selects an available PyTorch accelerator, otherwise CPU. It has no `--device` CLI option and does not automatically retry on CPU after an accelerator operation fails.

- **Unavailable device while loading:** the direct `load_policy(..., device="cpu")` API supports CPU loading. Confirm the checkpoint loads there before investigating accelerator support; this is not a trainer-device flag.
- **Generator/device mismatch:** put sampling tensors and their Torch generator on compatible devices. Moving the model alone is insufficient.
- **Accelerator out of memory:** reducing batch size is a diagnostic option for training tensors. Smaller fresh-model widths or horizons change the experiment; they do not reshape a loaded model.
- **Host memory pressure:** data loading materializes packed arrays, normalization concatenates training episodes, and prepared datasets can allocate more arrays. Reducing batch size may not address this stage. Identify whether failure occurs during preparation or training before changing settings.

Verify with the smallest relevant API/test path first, then a deliberately chosen integration run. These checks do not establish accelerator training support or a universal memory requirement.

## Checkpoint compatibility

| Symptom | Check and next step | Verify |
| --- | --- | --- |
| `Unsupported checkpoint format; expected version 1` | Confirm the file came from this policy writer. Legacy whole-model pickle and earlier experimental resume schemas are unsupported. | A supported file loads through `load_policy`; do not bypass the format check. |
| Missing component or strict weight-load error | Check producing revision, architecture, and weight keys/shapes. Follow [format-change guidance](guides/change-checkpoint-format.md) for an intentional migration. | Reconstruction and raw-space predictions match the intended policy. |
| Invalid normalization statistics/dimensions | Check saved means/scales and architecture compatibility. Do not refit statistics merely to make an old policy load. | Valid statistics match the model's state/action dimensions and original semantics. |
| `Configured dropout layer names do not match the model` | Check architecture changes or renamed modules. Saved dropout names must match reconstruction. | A compatible checkpoint restores layer probabilities and evaluation mode. |
| `Checkpoint flow_num_steps must be a positive integer` | Inspect the producing format/configuration; this field is validated even for MSE. | Supported payload contains a valid default and loads normally. |

A checkpoint is not exact training resumption: optimizer moments, RNG position, and training position are absent. See [save, load, and train further](guides/save-load-and-continue.md) rather than interpreting new counters as lost resume state.

## Randomness and repeatability

For differing results, compare code, data, backend, configuration, root seed, stream variations, and sampling shapes before calling the run nondeterministic.

| Symptom | Check | Verify |
| --- | --- | --- |
| Identical noise repeats every batch | Is the generator recreated with the same identity inside the loop? Retain it to advance. | Sequential draws advance; a deliberate recreation replays the start. |
| Changing monitoring affects training | Look for a shared/global RNG or changed draw routing. | Existing isolation checks pass; see [stream maintenance](guides/add-random-stream.md). |
| Final score differs from selection | Final evaluation has separate streams and possibly another budget. | Compare selected step and budgets in the final row. |
| CPU and accelerator results differ | Backend, operation, and library behavior can affect replay despite matching integer seeds. | Compare within a controlled backend before interpreting cross-device differences. |
| `Active dropout requires a caller-owned generator` | Pass the generator through the entire nested module path. | Training-mode dropout works and evaluation mode disables its draws. |

Use [randomness contracts](concepts/randomness.md) and repeat comparisons across seeds; one fixed seed does not measure robustness.

## Outputs and cleanup

| Symptom | Check and next step | Verify |
| --- | --- | --- |
| CSV files absent | Was `--log-csv False` supplied, or did failure precede logger creation? | Inspect expected files using [the results guide](guides/inspect-results.md). |
| W&B login prompt or offline error | Use `--log-wandb False` for local-only work. Offline mode alone is unsupported. | A chosen local run does not initialize W&B. |
| Videos absent | Check `num_video_episodes`, completed rollouts, and recording errors. CSV logging does not control recording. | Look in the correct selection/final video directory. |
| Checkpoint absent after success | `--save-checkpoints False` uses temporary storage that is removed on success; retention can remove older persistent files. | Compare flags and selected step with [retention rules](concepts/checkpoints.md). |
| Temporary checkpoints remain after failure | They are intentionally retained when the training body raises; cleanup itself can also fail. Preserve them while diagnosing. | Read the original exception and confirm which storage path remains. |
| Extra files/artifacts or cleanup warnings | Failed deletions remain tracked for retry during the process; retry state does not persist across runs. | Inspect actual local/remote state before choosing manual cleanup. |

`--save-checkpoints False` does not disable W&B uploads. Final CSV output can precede a later W&B reporting or cleanup failure. Use process status and traceback together with artifacts when deciding whether the run succeeded.

## Local versus CI

CI runs the [documented HW1 checks](tests-and-checks.md) on Linux with Python from `hw1/.python-version` and locked development dependencies.

1. Compare the failing job and command: documentation and HW1 checks use different environments.
2. Match the tested revision, dependency lock, working directory, and Python version locally.
3. Run the failing existing test selection, then the complete check sequence.
4. Read skip reasons: the accelerator-transfer test skips when hardware is unavailable; that does not establish GPU-training coverage.
5. If the same code passes locally, preserve the CI traceback and investigate platform/device or environment differences rather than changing the assertion blindly.

A useful handoff includes the exact command, revision, Python/backend details, traceback, relevant flags, and expected versus observed files. Exclude credentials from logs you share.

## CLI and sweep configuration

| Symptom | Next check | Expected result |
| --- | --- | --- |
| An old boolean flag is rejected | Replace bare flags and `--no-*` forms with explicit values such as `--log-wandb False`. | The [shared CLI](../reference/training.md#command-line-syntax) accepts the value. |
| A tuple argument is rejected | Quote a Python tuple literal, such as `--hidden-dims "(256, 256, 256)"`. | One tuple argument reaches the parser. |
| Sweep YAML is not found | Run from `hw1/` and use `conf/sweeps/mse.yaml` or `conf/sweeps/flow.yaml`. | W&B can read the chosen definition. |
| A changed YAML seems ignored | Register the edited definition and use its new sweep ID. | New trials use the new registered settings. |

For registration, trial execution, and results, follow [Run a hyperparameter sweep](guides/run-sweep.md).
