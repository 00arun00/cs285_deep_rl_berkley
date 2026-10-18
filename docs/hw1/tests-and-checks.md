# Run tests and checks

Use the existing checks to verify a code change before trying a real experiment. Start with [environment setup](setup.md); all commands below run from **`hw1/`** unless stated otherwise.

## Run the CI checks locally

After installing the locked development environment:

```bash
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check tests
uv run --no-sync pytest -v -rs
```

| Check | Expected result |
| --- | --- |
| Ruff lint | Exit successfully with no lint violations in `src` or `tests`. |
| Ruff formatting | Report that test files are already formatted. `--check` does not rewrite files. |
| pytest | Collected tests pass; `-rs` explains any skips. |

CI checks formatting for **tests only**. The lint command checks both source and tests. A missing test dependency is a setup problem: run `uv sync --locked --dev` before retrying.

## Narrow a test run

Run an entire test file:

```bash
uv run --no-sync pytest tests/test_checkpoint.py -v
```

Run one existing test:

```bash
uv run --no-sync pytest tests/test_checkpoint.py::test_cli_preserves_config_defaults -v
```

Select the flow variants in the checkpoint file:

```bash
uv run --no-sync pytest tests/test_checkpoint.py -k flow -v
```

Use a focused selection while investigating a failure, then run the full CI sequence before handing off the change. Check the collected/selected count: a misspelled node or filter can leave the intended behavior untested.

## Find the relevant test group

Files below are under `hw1/tests/`.

| Area | Existing files | Contracts checked |
| --- | --- | --- |
| Data | `test_data_boundaries.py` | Episode reconstruction, chunk boundaries, padding, indexing, and source-data isolation. |
| Numerical behavior | `test_numerical_correctness.py` | Example-weighted validation loss, mean episode-maximum rollout scores, and flow integration against a known solution. |
| Checkpoint loading and writing | `test_checkpoint.py`, `test_checkpoint_atomicity.py` | Policy round trips, fresh training from saved weights, invalid formats, and preserving files when serialization fails. |
| Retention | `test_checkpoint_retention.py`, `test_checkpoint_retention_properties.py` | Best/latest selection, ties, and cleanup retries after failures. |
| Randomness | `test_randomness.py`, `test_evaluation_streams.py` | Stable stream identities, isolation, replay, and separation of final evaluation. |
| Dropout | `test_generator_dropout.py`, `test_rng_modules.py`, `test_dropout_integration.py` | Explicit-generator behavior, evaluation mode, layer configuration, and checkpoint restoration. |
| Training outputs and control | `test_logging_controls.py`, `test_training_reporting.py`, `test_training_summary.py`, `test_evaluation_budgets.py`, `test_early_stopping.py` | Output switches, loss reporting, summary content, evaluation budgets, and plateau stopping. |

Some cases use Hypothesis to explore generated inputs; others parameterize policy families or output settings. A test file can therefore collect many cases.

## What passing tests establishes

The suite uses synthetic demonstrations and mocked data access, physics environments, and W&B services for training integration checks. It also exercises real tensor operations, policy serialization, and local file handling.

- It does not require downloading the Push-T dataset or contacting W&B.
- It writes temporary files and may write pytest, Python, and Hypothesis caches.
- An accelerator-transfer test skips when no accelerator is available. That skip does not mean CPU coverage failed; a pass does not establish GPU training coverage.
- Passing tests do not verify real environment rollouts, remote uploads, or a real-data training run. Use [the small local training test](test-runs.md) for the next level of integration.

## How CI relates to these commands

The [HW1 workflow](https://github.com/00arun00/cs285_deep_rl_berkley/blob/main/.github/workflows/hw1-ci.yml) runs the **HW1 checks** job on Ubuntu 24.04 for every pull request, regardless of target branch, and pushes to `main`.

It installs Python from `hw1/.python-version`, runs `uv sync --locked --dev`, and executes the three commands above from `hw1/`. Local success is useful evidence; CI also checks the Linux environment.

Documentation has a separate environment and workflow. For site changes, follow [documentation authoring](../contributing/index.md) from the repository root rather than installing site tools into HW1.
