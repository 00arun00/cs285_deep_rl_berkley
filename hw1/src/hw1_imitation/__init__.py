"""Train and evaluate action-chunking imitation policies for Push-T.

**Training workflow**

1. Prepare episode data, normalization statistics, and action windows with
   [data][hw1_imitation.data].
2. Choose a direct MSE or flow-matching policy in [model][hw1_imitation.model],
   then configure and execute a run through [train][hw1_imitation.train].
3. Measure validation loss and rollout scores with
   [evaluation][hw1_imitation.evaluation]. Save or reconstruct self-contained
   policies through [checkpoint][hw1_imitation.checkpoint].

**Supporting modules**

- [randomness][hw1_imitation.randomness] assigns stable random-stream identities;
  [rng_modules][hw1_imitation.rng_modules] routes generators through layers and
  manages explicit-RNG dropout.
- [logging_utils][hw1_imitation.logging_utils] formats summaries, writes metric
  files, and encodes videos. [early_stopping][hw1_imitation.early_stopping]
  tracks evaluation plateaus independently of checkpoint selection.
- [modal_train][hw1_imitation.modal_train] configures the optional Modal wrapper
  around the same training driver.

Import functionality from its defining module. This package initializer does
not re-export classes or launch training. Checkpoints reconstruct policies and
normalization; they do not resume optimizer state or training position.
"""
