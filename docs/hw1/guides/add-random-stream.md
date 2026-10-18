# Add or change a random stream

Use this guide when a new stochastic responsibility must stay independent of existing draws. If you only need another realization or another episode, use a variation or index instead. Read [stream identities and lifetimes](../concepts/randomness.md) first.

This is a procedure for a future code change. The documentation does not introduce a new stream. Source paths below are relative to **`hw1/src/hw1_imitation/`**; run checks from **`hw1/`** after [setup](../setup.md).

## 1. Choose the right coordinate

| Need | Use | Example |
| --- | --- | --- |
| An independent stochastic responsibility | New `StreamId`. | A new operation whose draw count should not affect training-loss noise. |
| Another realization of an existing responsibility | Existing stream with another `variation`. | Another train/validation split. |
| A stable peer within one responsibility | Existing stream with an explicit `index`. | Evaluation episode number. |
| Another experiment family | Another root seed. | Repeating an entire experiment. |

Do not allocate a new ID for every seed or episode. Do not reuse a training generator merely because it is already in scope.

## 2. Allocate a permanent ID

In `randomness.py`, inspect the registry before editing. It currently assigns IDs 1–10 and marks 11 as the next candidate.

1. Choose a descriptive responsibility name and an unused explicit integer.
2. Preserve every existing ID, including when renaming a member.
3. Never reuse a retired ID. Update the next-ID comment if applicable.
4. Preserve the identity encoding order `[stream, variation, index, root_seed]`, including the explicit zero index.

An illustrative future responsibility could use `NEW_RESPONSIBILITY = 11` only if 11 is still unused when the change is made. That symbol does not exist today.

## 3. Route configuration to the consumer

If users need to select realizations:

1. Add a named variation field with default zero to `TrainConfig` in `train.py`.
2. Validate it through `RandomStreamFactory.validate` with the new stream, rejecting invalid values before expensive work where possible.
3. Pass the field explicitly to the consumer; verify the dataclass-derived CLI flag and configuration reporting.

The factory accepts Python integer variations and indices in `[0, 2**32)`, excluding booleans. Current `TrainConfig` explicitly validates its listed stream variations, while final-evaluation variations are validated when used. Do not assume a newly added field is automatically checked by the existing validation loop.

Use `.torch(..., device=...)`, `.numpy(...)`, or `.seed(...)` according to the consumer. Match a Torch generator to the sampling device. Avoid global `manual_seed`, global NumPy draws, or hidden default generators in the new path.

## 4. Decide where the generator lives

| Desired behavior | Creation point |
| --- | --- |
| Advance across batches/epochs | Create once outside the loop and retain the object. |
| Replay draws for each complete assessment | Recreate with the same identity at the start of the assessment; retain within it. |
| Isolate peers from each other's draw counts | Create from a stable explicit index for each peer. |

Follow existing patterns: training loss advances through a run, validation loss restarts per pass, and each rollout episode has separate environment/policy identities. The factory does not advance indices automatically.

Pass the generator through every stochastic helper, including active dropout. If a nested module needs explicit RNG routing, inspect `rng_modules.py` rather than assuming ordinary `nn.Sequential` forwards a generator keyword.

## 5. Verify isolation and integration

Existing checks, run from `hw1/`:

```bash
uv run --no-sync pytest \
  tests/test_randomness.py tests/test_evaluation_streams.py \
  tests/test_rng_modules.py tests/test_dropout_integration.py -v
```

Expected result: existing identity, replay, default-RNG isolation, and monitoring-isolation checks still pass. Then run the [complete CI checks](../tests-and-checks.md).

For the new path, review whether the change demonstrates each contract below. Existing tests may not exercise a newly introduced responsibility; update relevant tests or propose additional coverage within the agreed scope of that implementation task.

- Same identity and compatible backend replay the initial draws.
- Reusing one generator advances it; unrelated draws do not advance it.
- Changing a variation changes only the intended responsibility's identity.
- Stable peer indices survive unrelated peers consuming more draws.
- Invalid coordinates fail, and generator/device mismatches are not hidden.
- Existing permanent seed identities remain unchanged.

## Common mistakes

- Recreating a training generator in every batch unintentionally repeats its noise.
- Changing stream numbers to match declaration order breaks established identities.
- Using loop position as an index after reordering peers changes their identities.
- A passing replay test alone does not prove isolation from global RNG state.
- A fixed seed does not promise identical results across devices or changed sampling operations.

See [RandomStreamFactory][hw1_imitation.randomness.RandomStreamFactory] for the current API.
