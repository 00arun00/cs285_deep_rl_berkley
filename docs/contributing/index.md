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

The site is written to `site/`. Documentation CI runs the strict build and pilot checks on every PR, on pushes to `main`, and on published releases. A separate trusted workflow publishes successful artifacts. Documentation dependencies are isolated from each training project's environment.

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
The first documentation-quality pilot is
[`Normalizer`][hw1_imitation.data.Normalizer]. Other classes retain their
existing source documentation, including
[`TrainConfig`][hw1_imitation.train.TrainConfig].

Write Google-style class and member docstrings beside the code. Use `Attributes`
for field meanings, `Args` for parameters, and `Returns` and `Raises` where
applicable. Names, annotations, signatures, and defaults come from source.
Ordinary `#` comments appear in source disclosure but are not field descriptions.
For an unannotated property return, use a named Google entry such as
`count (int): Number of entries.` Do not change executable annotations merely
for documentation. Inspection is disabled, so builds do not import training code.

Class coverage includes explicitly defined methods, properties, constructors,
dunders, and private helpers; inherited members and synthesized constructors
are not repeated. Private helpers are marked as implementation details.
The checker reports function-local classes and detectable dynamic assignments
as exceptions. Arbitrary runtime-created state remains a static-analysis limit.

Generated pages are canonical for Python-object links. Existing comparison
pages retain their URLs and anchors but set `skip_local_inventory: true`, so
their duplicate renderings do not register symbolic targets. Link from authored
guides with `[Normalizer][hw1_imitation.data.Normalizer]`. Distinct pages can
still appear separately in search during this comparison phase.

### Validate the pilot

Build first, then run the documentation checker using the documentation Python
and an existing Ruff executable:

```bash
uv run --no-sync zensical build --strict
RUFF_BIN=/path/to/ruff uv run --no-sync python scripts/check_docs.py
```

Ruff 0.14.10 is already declared and locked by HW1; use an existing installation
without installing the training environment. The checker never installs tools. Documentation CI installs only standalone
Ruff 0.14.10, then runs this same checker after the strict build.
`docs/ruff.toml` selects Google-style docstring checks. It lints the exact source
of the pilot class, leaving missing-documentation checks to the structural
checker rather than Ruff's public-only rules.

`scripts/check_docs.py` reconciles source classes against Griffe extraction and
actual native-generated HTML anchors. It enforces class/member docstrings,
attribute and parameter descriptions, stale names, empty descriptions, and
rendered member anchors for `Normalizer`. The JSON report records coverage gaps
for all other classes without failing the pilot. Use `--report PATH` to save it,
and optionally `--baseline PATH` to verify `data.py` has an unchanged executable
AST after docstrings are removed. These checks do not prove semantic accuracy;
source review and desktop/mobile interaction review remain required.

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
