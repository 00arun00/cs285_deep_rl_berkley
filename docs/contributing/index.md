# Write documentation

## Required standards

Before completing any code change, review affected source docstrings,
guides, and command examples and update them to reflect the implementation.

### Module-level documentation

Review the module-level docstring whenever adding or updating a module.
Update it as needed to reflect the module's purpose, workflow, entry points,
and limitations.

Every Python module must have a non-empty module docstring.
Use the opening triple-quoted string, not ordinary `#` comments, so the
documentation tools can extract it.

Write it as an orientation to the module:

- State its purpose and responsibilities.
- Explain the main workflow or how its components fit together.
- Link to the relevant entry points using generated API references.
- Describe important ownership rules, side effects, and limitations
  where applicable.

Scale the detail to the module. Keep individual parameter contracts in
the corresponding callable docstrings.

The existing [checkpoint module][hw1_imitation.checkpoint] provides a
reference example:

```python title="hw1/src/hw1_imitation/checkpoint.py · existing module docstring"
"""Save and load self-contained policies for inference or a new training run.

**Checkpoint contents**

- Architecture configuration and learned model weights.
- Runtime training configuration: per-layer dropout probabilities.
- Observation/action normalization statistics and default flow inference steps.

Use [save_policy][hw1_imitation.checkpoint.save_policy] to write a checkpoint,
then [load_policy][hw1_imitation.checkpoint.load_policy] to reconstruct it without
the original dataset. Weights are stored on CPU; configuration is plain data.

Loading does not resume a training session. No optimizer, training position,
or saved RNG state is restored. Further training uses a new optimizer and
caller-owned randomness; saved dropout settings can be explicitly overridden.
"""
```

This explains what the module stores, how to use it, and the distinction
between loading a policy and resuming training.

### Classes, fields, and callables

Use Google-style docstrings beside the code:

| Element | Required documentation |
| --- | --- |
| Class | Purpose, behavior, and relevant usage constraints. |
| Class or stored instance field | Meaning in `Attributes`; ordinary implementation comments do not substitute for field documentation. |
| Function or explicitly defined method | Purpose and relevant behavioral contract, including private helpers and constructors. |
| Parameter | Meaning and constraints in `Args`. |
| Returned or yielded value | Meaning in `Returns` or `Yields`, where applicable. |
| Exception | Relevant failure conditions in `Raises`. |
| Side effect | Explain meaningful effects such as writing files, changing state, or contacting an external service. |

Type annotations and signatures do not replace descriptions. Inherited
and imported/re-exported objects do not require duplicate documentation.

For a complete existing example, read
[load_policy][hw1_imitation.checkpoint.load_policy]. Its docstring explains
the load sequence, ownership of randomness, device behavior, parameters,
returned objects, and failure conditions.

This excerpt shows how it documents returned values precisely:

```python title="checkpoint.load_policy · existing docstring excerpt"
    Returns:
        model: Loaded policy in evaluation mode on `device`.
        normalizer: Saved state/action statistics with independent copied storage.
        inference_config: Dictionary containing the validated `flow_num_steps` default.
```

Use these existing examples as references for clarity and completeness;
do not copy details that do not apply to the code being documented.

### Guides and command examples

When invocation or behavior changes, review the corresponding operational
guides and examples.

- State the working directory for commands.
- Use language-labeled code blocks.
- Explain relevant inputs, outputs, and side effects.
- Use relative Markdown links between authored pages.
- Link Python objects to their generated API documentation.

Keep operational instructions in guides and API meanings in source
docstrings. Apply updates within the approved scope; identify additional
changes requiring approval before proceeding.

## Required checks

All production Python under `hw1/src` is covered by the documentation
gate, including new modules and small helpers. Tests under `hw1/tests`
are outside that gate.

Before completing affected changes:

1. Review documentation against the implementation.
2. Build the documentation with strict validation.
3. Run the whole-source documentation gate.

From the repository root, after installing documentation dependencies
and Ruff as described below:

```bash title="Build and validate documentation"
uv run --no-sync zensical build --strict
uv run --no-sync python scripts/check_docs.py --all --report docs-coverage.json
```

See [Validate production documentation](#validate-production-documentation)
for tooling setup and detailed enforcement.

CI runs these checks too. Ordinary lint and application tests are not
substitutes:

- Ruff checks docstring style using the documentation configuration.
- The documentation gate checks required structured descriptions and
  reconciles source objects with generated API output.
- Human review checks accuracy and whether module overviews, guides,
  and examples adequately explain the implementation.

The gate requires a non-empty module docstring; it cannot judge whether
that overview explains the module well.

Report which checks passed and any that could not be completed.

## Preview and build

From the repository root:

```bash title="Start a local preview"
uv sync --locked
uv run --no-sync zensical serve
```

```bash title="Build the static site"
uv run --no-sync zensical build --strict
```

The site is written to `site/`. Documentation CI runs the strict build and whole-source documentation checks on every PR, on pushes to `main`, and on published releases. A separate trusted workflow publishes successful artifacts. Documentation dependencies are isolated from each training project's environment.

## Add a page

Add a Markdown file under `docs/`. Navigation discovers new pages automatically. Change the nearest `.nav.yml` only to control labels or order; keep its `glob: "*"` entry so unlisted pages still appear.

Use relative Markdown links such as `[Test runs](../hw1/test-runs.md)`. Keep operational guides in `docs/hw1/` and API pages in `docs/reference/`.

## Write code examples

Label fenced blocks with their language for syntax highlighting. Code blocks have a copy button. Add a filename with `title`, optional line numbers with `linenums`, and emphasized lines with `hl_lines`:

````markdown
```python title="example.py" linenums="1" hl_lines="2"
from pathlib import Path
output = Path("exp")
print(output)
```
````

```python title="example.py" linenums="1" hl_lines="2"
from pathlib import Path
output = Path("exp")
print(output)
```

## Generate API documentation

The native `api-autonav` integration discovers production modules beneath
`hw1/src/hw1_imitation` and creates the **Generated API** section under `/api/`.
Adding a class in that package requires no Markdown page or navigation entry.
Discovery includes undocumented objects; generated does not mean reviewed.
All production modules, classes, own members, and standalone functions under
`hw1/src` are covered by the documentation gate. Tests under `hw1/tests` are
outside this scope. Imported/re-exported and inherited objects are not counted
as newly authored APIs.

Write Google-style docstrings beside the code. Use `Attributes` for class and
stored instance fields, `Args` for parameters, and `Returns` or `Yields` for
outputs. Document exception behavior in `Raises`. Keep meanings in Python
source; shared templates derive signatures, types, and values from metadata.

Ordinary `#` comments appear in source disclosure but are not field descriptions.
For an unannotated property return, use a named Google entry such as
`count (int): Number of entries.` Do not change executable annotations merely
for documentation. Inspection is disabled, so builds do not import training code.

Class coverage includes explicitly defined methods, properties, constructors,
dunders, and private helpers; inherited members and synthesized constructors
are not repeated. Private helpers are marked as implementation details.

Generated pages are canonical for Python-object links. Existing comparison
pages retain their URLs and anchors but set `skip_local_inventory: true`, so
their duplicate renderings do not register symbolic targets. Link from authored
guides with `[Normalizer][hw1_imitation.data.Normalizer]`. Distinct pages can
still appear separately in search during this comparison phase.

### Validate production documentation

The [required checks](#required-checks) use the same whole-source gate as CI.

Install documentation dependencies from the repository root:

```bash title="Install documentation dependencies"
uv sync --locked
```

Install Ruff:

```bash title="Install documentation lint tooling"
uv tool install ruff==0.14.10
```

After the strict build, run:

```bash title="Check production API documentation"
uv run --no-sync python scripts/check_docs.py --all --report docs-coverage.json
```

Ruff must be available on `PATH`. Alternatively, pass the path to an
existing executable through `--ruff`. Ordinary Ruff invocation alone
does not run the structured-documentation and generated-API checks.

The default without `--all` is also whole-source enforcement. CI explicitly
passes `--all` through `scripts/build_docs.py` before preparing an artifact.
Ruff checks every production Python file; no training environment is installed.
For a focused authoring check, use `--pilot qualified.Class`. This opt-in mode
is not the CI gate. Its optional `--baseline PATH` compares that class's module
against pre-edit source with docstrings removed.

The gate rejects missing or empty module/class/member/field/parameter/output
documentation, stale fields and parameters, missing or empty structured raises,
and missing/duplicate native API anchors or inventory entries. It reconciles
AST and native class/function inventories, including private definitions.
Explicit named raises outside `try` blocks require descriptions; caught,
rethrown, dynamic, and dependency exceptions still need semantic review.

Function-local helpers are checked in source and reported as exceptions to
native discovery: they have no independently generated API object or link.
Function-local classes and arbitrary runtime-created attributes need explicit
review. Structural checks cannot establish behavioral accuracy, validate every
possible runtime object, or replace desktop/mobile rendered review.

## Maintain the theme

Navigation, search, code controls, and the light/dark toggle are native Zensical features. `docs/stylesheets/theme.css` customizes the palette, system font stacks, spacing, and borders. The shared mkdocstrings templates in `docs-overrides/python/material/` combine structured Python attribute docstrings with source annotations and defaults when `options.extra.documented_fields` is enabled. Generated modules enable this strategy globally; the preserved comparison page opts in locally. A small function wrapper omits synthesized methods without source lines and labels private methods. Parameter, return, and exception sections use native lists to remain readable on mobile. Other rendering remains native. `docs-theme/main.html` adds a build/channel announcement to published pages using the native announcement block. There is no custom JavaScript.

Check both color modes and narrow screens when changing that stylesheet.

## Publish documentation

The public site uses GitHub Pages at
`https://00arun00.github.io/cs285_deep_rl_berkley/`.

| Path | Source | Lifecycle |
| --- | --- | --- |
| `/latest/` | Latest successful build of the current `main` commit | Updated after merges |
| `/v1.0.0/` | Published release tag `v1.0.0` | Immutable snapshot |
| `/stable/` | Highest published semantic release | Per-page redirects preserve deep links |
| `/previews/pr-38/` | Successful PR build (GitHub's merge snapshot) | Removed on close/merge or expiry; manual rebuilds can restore it |
| `/` | Redirect to `/latest/` | Permanent entry point |

The release workflow supports published, non-prerelease tags named
`vMAJOR.MINOR.PATCH`. Merely pushing a tag does not publish docs: publish its
GitHub Release. The tagged commit must contain the documentation build workflow
and its locked tooling. Older versions remain available in the version selector;
PR previews are excluded from that selector. Each published page identifies its
channel and actual build commit. PR build records also retain the PR head and base
commit. Preview pages carry `noindex, nofollow` metadata.

### One-time setup and the first deployment

1. In repository **Settings → Pages**, set **Source → GitHub Actions**.
2. Enable Actions for this repository if GitHub disabled them when it was forked.
3. In **Settings → Environments → github-pages**, restrict deployment branches to
   `main`. The publishing workflow runs in the default-branch context even when
   handling a release or PR build. No personal access token is required, and
   default workflow token permissions may remain read-only.
4. Merge the reviewed publishing workflow into `main`. The resulting main build
   publishes `/latest/`. Before that merge, the infrastructure PR can validate
   and upload its artifact but cannot trigger its own trusted publisher.
5. Verify the **Publish documentation** run and the live site. Existing open PRs
   need a new successful Documentation build or a manual publisher dispatch.

The publishing workflow and scripts always come from `main`; it never checks
out or executes PR or release source in its privileged job. The read-only build
job uploads static output. The publisher validates the workflow/run, current PR
head or release tag, metadata and archive paths before consuming it. A PR cannot
choose its deployment directory. For PRs, GitHub's recorded revision of the
locally called build workflow must match the artifact's merge SHA. Builds without
that provenance must be rebuilt with the updated workflow. Pages publication is serialized with queued
runs. Automatic builds for superseded PR heads are skipped. A late build can
restore a closed PR preview; expiry cleanup handles this accepted race. All output is retained on the
machine-managed `gh-pages` branch; avoid editing that branch manually.

### Rebuild and publish a preview

Automatic PR builds use GitHub's synthetic merge checkout: the PR changes combined
with its target branch. Main builds use the current main commit; release builds
use the tagged commit. Build success and publication are separate operations.
Publication is asynchronous and does not need to finish before merging.

Every successful PR build processed by the publisher records a snapshot named
`RUN_ID-ATTEMPT`, for example `123456789-1`. Find this ID in the preview comment or
the publisher's **Validate artifact and assemble all published versions** log.
Unapproved fork builds also get records; their IDs appear in that log. Records
live under `.docs-state/snapshots/` on `gh-pages`, independently of expiring build
artifacts and preview pages. A failed build does not produce a rebuildable record.

To rebuild that exact source, including after the PR closes or merges:

1. Open **Actions → Rebuild and publish preview → Run workflow**.
2. Choose branch **main** and enter the snapshot ID.
3. Run the workflow. This explicitly approves publication of that snapshot,
   including content from an external fork.

The workflow checks out the recorded merge commit, installs its locked build
requirements, and runs the documentation checks and build. It publishes to the
same `previews/pr-N/` URL and updates the PR comment. This is a fresh build of the
recorded source, not a new merge against today's main. If the commit can no longer
be retrieved or its dependencies no longer build, the operation fails without
substituting another commit. Old builds without snapshot records are unsupported.

Historical source executes in a read-only job without publishing credentials.
The publisher runs separately using trusted main scripts. The manual workflow
has no 30-day rerun-window restriction; it still depends on source availability.

### Fork approval and retries

PRs from branches within this repository publish automatically, even when this
repository itself is a fork. PRs from an external fork require publication
approval regardless of their target branch. Approval applies to the exact PR
snapshot: retries of that snapshot reuse approval, but a changed head or merge
snapshot needs fresh approval. GitHub's approval to run a fork build is separate.

An ordinary **Re-run all jobs** retries the original build and does not grant
fork-publication approval. GitHub limits native reruns to 30 days after the
original run. Use **Rebuild and publish preview** for older recorded snapshots.

To approve or retry publication of an existing successful artifact without
rebuilding, open **Publish documentation → Run workflow** on **main**, select
`publish`, and enter its Documentation run ID. The latest attempt must have
succeeded. This action also approves its exact fork snapshot and can restore a
closed PR's preview. Artifacts expire after 14 days; use the recorded snapshot
rebuild when the artifact is gone. An obsolete main build cannot replace current
main documentation. To retry a failed manual rebuild's publisher, rerun its failed
jobs; the publisher can reuse that workflow run's already-successful build artifact.

A failed build leaves the last published preview intact. A failed deployment does
not advance the preview's publication timestamp. The publisher retains pending
state so a later deployment can finish it. Failed PR-comment updates are retained
for retry independently of the website deployment. A comment API failure emits
a warning and does not prevent deployment or updates to other PR comments.

### Remove and expire previews

Closing or merging a PR queues automatic removal of its preview. Cleanup checks
that the PR is still closed before proceeding. A later explicit rebuild can
restore the preview; a late automatic publication can also restore it. Both then
follow the same expiry rule.

A daily cleanup runs at **03:17 UTC**. It removes previews whose last successful
publication is more than **30 days** old, whether their PRs are open or closed.
Set the repository Actions variable `DOCS_PREVIEW_RETENTION_DAYS` to a positive
integer to change the retention period. PR creation time is not used. Successful
publication resets the clock; failed builds and deployments do not.

For an immediate expiry sweep, open **Publish documentation → Run workflow** on
**main**, select `cleanup`, and leave the run ID empty. This uses exactly the daily
cleanup logic. To retry immediate removal after a failed close/merge event, rerun
that event's failed **Publish documentation** job; an expiry sweep intentionally
does not remove a recent preview merely because its PR is closed.

Cleanup removes preview files and updates their comments. Main/release versions,
snapshot records, and approvals are retained. Removal from the current site does
not erase Git history. Existing previews without timestamps get one retention
period from their first successful migration deployment. If a deployment's final
bookkeeping fails, a subsequent deployment recovers its pending state before
expiry is considered.

### Publishing implementation

`mike` is pinned to the Zensical-compatible fork in the optional `publishing`
dependency group. It manages the version inventory, aliases, redirects and git
commits. `scripts/build_docs.py` supplies the URL prefix and announcement metadata,
then runs the strict build and existing checker. `scripts/publish_docs.py` uses
the pinned mike API to publish already-built files; it deliberately does not use
`mike deploy`, which would execute a build in the privileged job. Durable snapshot,
approval, pending-deployment and publication records live in `.docs-state/` on
`gh-pages`; these internal records are excluded from the Pages artifact. Updates to mike
must revalidate that API. The publishing job alone receives branch-write, Pages,
OIDC, artifact-read and PR-comment permissions.
