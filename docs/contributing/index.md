# Write documentation

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

After building, run the same gate used by CI:

```bash title="Check production API documentation"
uv tool install ruff==0.14.10
uv run --no-sync python scripts/check_docs.py --all --report docs-coverage.json
```

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
| `/previews/pr-38/` | Successful PR build (GitHub's merge snapshot) | Updated while open; removed when closed |
| `/` | Redirect to `/latest/` | Permanent entry point |

The release workflow supports published, non-prerelease tags named
`vMAJOR.MINOR.PATCH`. Merely pushing a tag does not publish docs: publish its
GitHub Release. The tagged commit must contain the documentation build workflow
and its locked tooling. Older versions remain available in the version selector;
PR previews are excluded from that selector. Each published page identifies its
channel and source commit. Preview pages carry `noindex, nofollow` metadata.

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
choose its deployment directory. Pages publication is serialized with queued
runs, and stale/closed PR builds are skipped. All output is retained on the
machine-managed `gh-pages` branch; avoid editing that branch manually.

### Approve an external-fork preview or retry publication

PRs whose head branch lives in this repository publish automatically. The fact
that this repository is itself a fork does not change that rule. External-fork
builds are checked but are not automatically published: their HTML/JavaScript
would otherwise be hosted on our Pages origin without review.

After reviewing an external-fork PR:

1. Open its successful **Documentation** run and copy the numeric run ID from
   the URL (`.../actions/runs/123456789`).
2. Open **Actions → Publish documentation → Run workflow**.
3. Choose branch **main** and enter that run ID.

This manual dispatch approves publication of that exact current PR head. A new
push needs a new approval. Closed or outdated builds are rejected. A repository
maintainer with permission to run workflows performs this approval; approving
GitHub's fork *build* prompt alone does not approve publication.

The same dispatch retries a failed publication for a successful main, release,
or PR build. Artifacts expire after 14 days; rerun the build if its artifact has
expired. Successful PR deployment updates one preview-link comment with the
source SHA. Closing/merging the PR removes the preview and updates that comment.
A failed deploy leaves the previous live site in place; the retained `gh-pages`
state lets a retry finish publication. Re-running a release preserves its original
bytes, and moving a published release tag is rejected rather than silently
rewriting history. Restoring a deliberately removed preview requires a new build
or an approved dispatch for a currently open PR.

### Publishing implementation

`mike` is pinned to the Zensical-compatible fork in the optional `publishing`
dependency group. It manages the version inventory, aliases, redirects and git
commits. `scripts/build_docs.py` supplies the URL prefix and announcement metadata,
then runs the strict build and existing checker. `scripts/publish_docs.py` uses
the pinned mike API to publish already-built files; it deliberately does not use
`mike deploy`, which would execute a build in the privileged job. Updates to mike
must revalidate that API. The publishing job alone receives branch-write, Pages,
OIDC, artifact-read and PR-comment permissions.
