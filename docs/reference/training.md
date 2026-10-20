# Training configuration

These defaults come from the source. For a small local run, use the [test-run recipe](../hw1/test-runs.md).

## Command-line syntax

Run commands from **`hw1/`**. Individual runs and W&B sweep trials use the same
`hw1_imitation.train` parser. The Modal training entry point also shares this parser.

```bash title="hw1 · inspect available options"
uv run --no-sync python -m hw1_imitation.train --help
```

| Kind | Example | Accepted form |
| --- | --- | --- |
| Boolean | `--log-wandb False` | Explicit `True` or `False`; bare flags and the old `--no-*` forms are not accepted. |
| Tuple | `--hidden-dims "(256, 256, 256)"` | One quoted Python tuple literal; a singleton is `"(256,)"`. |
| Fixed-size tuple | `--video-size "(256, 256)"` | Exactly two integer dimensions. |
| Optional value | `--init-from None` | Literal `None` clears an optional value. |
| Number | `--lr 0.0003` | Supply a value of the field's declared type. |
| Path | `--data-dir "data with spaces"` | Quote shell arguments containing spaces. |

Hyphens and underscores are accepted in option names. Values can follow a space
or `=`, such as `--log_wandb=False`. Omitted fields retain their defaults;
configuration validation still runs after parsing. Replace old space-separated
tuple elements with one quoted literal.

For repeated trials, see [Run a hyperparameter sweep](../hw1/guides/run-sweep.md)
and the [sweep configuration reference](sweeps.md). The YAMLs override selected
fields; they do not change the defaults below.

## Training fields

::: hw1_imitation.train.TrainConfig
    options:
      skip_local_inventory: true
      show_if_no_docstring: true
      merge_init_into_class: false
      show_labels: false
      show_symbol_type_heading: false
      separate_signature: false
      docstring_section_style: table
      filters:
        - "!^__init__$"
      show_docstring_attributes: false
      extra:
        documented_fields: true
