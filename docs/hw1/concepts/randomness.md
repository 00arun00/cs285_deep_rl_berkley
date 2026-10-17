# Random streams and reproducibility

The trainer gives each stochastic responsibility its own identity. Changing how often validation runs should not consume the random draws used by training loss. This separation is implemented by `RandomStreamFactory`, not by repeatedly setting a global seed.

## An identity has four coordinates

| Coordinate | Role | Example |
| --- | --- | --- |
| Root seed | Identifies the experiment family. | `--seed 42`. |
| Stream ID | Names a permanent responsibility. | `TRAIN_LOSS`. |
| Variation | Selects another realization of that responsibility. | `--train-loss-variation 1`. |
| Index | Identifies a peer within the stream. | Rollout episode 3. |

The implementation passes `[stream ID, variation, index, root seed]` to NumPy `SeedSequence`, keeping the field order and explicit zero index stable. NumPy generators use PCG64; Torch generators receive a derived unsigned 64-bit seed.

The factory stores no advancing generator state. Every request creates a **fresh** generator at its starting position. Keep a generator object to advance through draws; request the same identity again to replay from the beginning.

## See replay versus advancement

This illustrative snippet draws numbers only; it does not train or evaluate a policy:

```python
from hw1_imitation.randomness import RandomStreamFactory, StreamId

streams = RandomStreamFactory(root_seed=42)
a = streams.numpy(StreamId.TRAIN_LOSS, variation=0)
first = a.normal(size=3)
second = a.normal(size=3)  # Continues the same generator.

b = streams.numpy(StreamId.TRAIN_LOSS, variation=0)
replayed_first = b.normal(size=3)  # Replays first, not second.
c = streams.numpy(StreamId.TRAIN_LOSS, variation=1)  # Another realization.
```

Creating `a` again before every batch would restart its noise sequence. The trainer deliberately retains the training-loss generator instead.

## Know each stream's lifetime

| Responsibility | Permanent ID | Lifetime in the trainer |
| --- | --- | --- |
| Episode split | `DATA_SPLIT = 1` | Created for the train/validation split. |
| Fresh model initialization | `MODEL_INIT = 2` | CPU generator used while building the model. |
| Training loader | `TRAIN_LOADER = 3` | Attached to the loader; advances across iterations. |
| Validation loader | `VALIDATION_LOADER = 4` | Attached to the ordered loader; retained across passes. |
| Training loss | `TRAIN_LOSS = 5` | One generator advances across batches and epochs, including noise, times, and active dropout. |
| Validation loss | `VALIDATION_LOSS = 6` | Recreated for each complete validation pass; reused within that pass. |
| Selection environment and policy | `EVAL_ENV = 7`, `EVAL_POLICY = 8` | Fresh identity per episode index on every selection call. |
| Final environment and policy | `FINAL_EVAL_ENV = 9`, `FINAL_EVAL_POLICY = 10` | Separate identities per episode in the final call. |

An ordered validation loader still has its own generator for loader internals. It is distinct from the generator passed to the stochastic validation loss.

During a rollout, the policy generator advances across chunks in that episode. The next episode uses its own index. A longer earlier episode therefore does not consume the next episode's policy draws. Each evaluation call starts episode indexing at zero; periodic calls reuse matching identities.

## Vary one source deliberately

With the root seed fixed, changing `eval_env_variation` changes selection-environment seeds while leaving training-loss and model-initialization identities unchanged. Changing `eval_policy_variation` changes policy sampling streams instead.

Keep this distinction when interpreting a comparison:

- MSE inference in evaluation mode is deterministic for fixed inputs and weights, so a policy-sampling variation may have no effect.
- Flow inference consumes noise, even in evaluation mode.
- Active dropout consumes the supplied generator during training; `eval()` disables dropout draws.
- Changing an algorithm's draw order or tensor shapes can change its results despite an unchanged identity.

## What reproducibility does and does not promise

| Stable contract | Boundary |
| --- | --- |
| A repeated identity derives the same starting seed under unchanged derivation behavior. | Distinct identities are not mathematically guaranteed never to collide. |
| Factory calls do not mutate global NumPy or Torch RNG state. | Other code can still use global RNGs; use the explicit-generator paths consistently. |
| Training and monitoring have separate generators. | Library versions, device backends, operations, and sampling shapes can affect numerical replay. |
| Stream IDs are explicitly assigned. | Reordering ID declarations is harmless; renumbering or reusing an ID changes established identities. |

Match a Torch generator's device to the operation that consumes it. The device does not change the derived integer seed, but CPU and accelerator draws need not be identical.

Root seeds must be nonnegative Python integers. Variations and indices must be Python integers in `[0, 2**32)`; booleans are rejected. An index does not increment automatically.

Policy checkpoints do not store generator states. Loading weights therefore cannot resume an interrupted random trajectory. See [checkpoint limits](checkpoints.md) and the [RandomStreamFactory API][hw1_imitation.randomness.RandomStreamFactory].
