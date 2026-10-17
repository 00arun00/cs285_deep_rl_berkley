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

The site is written to `site/`. The documentation workflow runs the same strict build when docs, tooling, or documented Python source changes; it does not publish the site. Documentation dependencies are isolated from each training project's environment.

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

Navigation, search, code controls, and the light/dark toggle are native Zensical features. `docs/stylesheets/theme.css` customizes the palette, system font stacks, spacing, and borders. The shared mkdocstrings templates in `docs-overrides/python/material/` combine structured Python attribute docstrings with source annotations and defaults when `options.extra.documented_fields` is enabled. Generated modules enable this strategy globally; the preserved comparison page opts in locally. A small function wrapper omits synthesized methods without source lines and labels private methods. Parameter, return, and exception sections use native lists to remain readable on mobile. Other rendering remains native. There is no custom JavaScript.

Check both color modes and narrow screens when changing that stylesheet.
