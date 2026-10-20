# Documentation standards

Before completing any code change, read
[Documentation authoring](../docs/contributing/index.md).

## Review affected documentation

- Review module-level docstrings whenever adding or updating a module.
  Update them as needed to reflect its purpose, workflow, entry points,
  and limitations.
- Review class, field, function, and method documentation when their
  contracts change. Small helpers and private methods are included.
- Review affected guides and command examples when invocation or
  behavior changes.
- Keep API meanings in source docstrings and operational instructions
  in authored guides. Use generated API references for Python objects.

Follow the standards and existing-code examples in the authoring guide.
Apply updates within the approved scope; identify additional changes
requiring approval before proceeding.

## Validate before completion

Run the strict documentation build and whole-source documentation gate
described in the authoring guide. Ordinary Ruff checks and application
tests do not replace these checks.

Review prose and examples for accuracy. Automated checks cannot establish
that documentation correctly describes behavior.

Report which checks passed and any that could not be completed.
