# Python reference

Signatures, type annotations, docstrings, and expandable source are generated directly from `hw1/src/hw1_imitation` at build time.

- [Training configuration](training.md)
- [Sweep configuration](sweeps.md): authored reference for W&B YAML definitions.
- [Policies](policies.md)
- [Early stopping](early-stopping.md)

The documentation builder reads Python source without importing the training modules. Ordinary implementation comments remain visible in the expandable source; structured API prose comes from docstrings.
