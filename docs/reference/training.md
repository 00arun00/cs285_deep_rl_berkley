# Training configuration

These defaults come from the source. For a small local run, use the [test-run recipe](../hw1/test-runs.md).

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
