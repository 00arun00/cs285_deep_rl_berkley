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

The site is written to `site/`. Documentation CI runs the rendering tests and strict build on every PR and on pushes to `main`. Whole-source completeness enforcement will be enabled after the source documentation is complete. Documentation dependencies are isolated from each training project's environment.

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

After building, run the completeness checker while authoring:

```bash title="Check production API documentation"
uv tool install ruff==0.14.10
uv run --no-sync python scripts/check_docs.py --all --report docs-coverage.json
```

The default without `--all` is also whole-source enforcement. Whole-source CI enforcement is deferred until the source documentation is complete.
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

Navigation, search, code controls, and the light/dark toggle are native Zensical features. `docs/stylesheets/theme.css` customizes the palette, system font stacks, spacing, and borders. The shared mkdocstrings templates in `docs-overrides/python/material/` combine structured Python attribute docstrings with source annotations and defaults when `options.extra.documented_fields` is enabled. Generated modules enable this strategy globally; the preserved comparison page opts in locally. A small function wrapper omits synthesized methods without source lines and labels private methods. Parameter, return, and exception sections use native lists to remain readable on mobile. Other rendering remains native. There is no custom JavaScript.

Check both color modes and narrow screens when changing that stylesheet.

