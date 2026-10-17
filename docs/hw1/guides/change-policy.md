# Add or change a policy

Use this guide when changing policy architecture, loss, or sampling behavior. It describes a future implementation task; the supported policy types remain `mse` and `flow`.

Read the [policy reference](../../reference/policies.md), [data pipeline](../concepts/data-pipeline.md), and [randomness guide](../concepts/randomness.md) first. Source paths below are relative to **`hw1/src/hw1_imitation/`**; run checks from **`hw1/`** after [setup](../setup.md).

## 1. Preserve the public tensor contract

| Method | Inputs | Required behavior |
| --- | --- | --- |
| `reset_parameters(generator)` | Caller-owned initialization generator. | Initialize all parameters through that generator; work with the builder's CPU allocation path. |
| `compute_loss(state, action_chunk, generator=...)` | State `(B, S)` and target actions `(B, H, A)`, in normalized coordinates. | Return a differentiable scalar batch-mean loss. |
| `sample_actions(state, generator=..., num_steps=...)` | State `(B, S)` and caller-owned sampling generator. | Return normalized action chunks `(B, H, A)`. |

`B` is batch size, `S` state dimension, `H` chunk size, and `A` action dimension. The base class exposes these dimensions through `PolicyConfig`.

Normalization and denormalization belong to the caller. The evaluator validates action chunks, clips them to environment bounds, and executes a chunk before requesting another. Do not embed a second normalization step or change the action meaning without coordinating the whole pipeline.

Loss reduction is part of the contract: the logger and validator weight batch means by example counts. Returning a sum would silently distort those metrics.

## 2. Trace construction and configuration

For a new family in `model.py`:

1. Implement the `BasePolicy` methods and validate the architecture it accepts.
2. Add its name to `PolicyType` and dispatch it in `construct_policy`.
3. Decide whether `PolicyConfig` needs new architectural fields. That configuration is serialized; coordinate any compatibility change with [checkpoint-format work](change-checkpoint-format.md).
4. Verify `build_policy`, CLI selection, and checkpoint loading all reach the new constructor.

`build_policy` constructs on the meta device, allocates empty CPU storage, calls `reset_parameters` with a supplied CPU generator, then moves the model to the target device. Avoid constructor logic that requires materialized tensors or reads uninitialized parameter values.

Direct `construct_policy` uses ordinary layer construction and does not promise global-RNG isolation. Preserve the explicit-generator builder and the loader's meta-device reconstruction paths.

## 3. Route randomness and respect mode

- Use the supplied generator for stochastic loss sampling, action sampling, and active dropout.
- If introducing an independent responsibility outside the policy's existing loss/sampling streams, follow [Add a random stream](add-random-stream.md).
- Keep generator and sampling tensor devices compatible.
- Ensure `train()` and `eval()` correctly enable/disable dropout through nested modules.
- Do not assume every sampler disables gradients. The existing MSE sampler does not; callers use `torch.no_grad()` for inference.

Existing explicit-RNG layers live in `rng_modules.py`. Their module names identify serialized dropout probabilities. Ordinary PyTorch dropout is not included in that configuration helper; swapping layer types can change both RNG routing and checkpoint behavior.

## 4. Check trainer and evaluator integration

| Integration point | What to verify |
| --- | --- |
| Fresh model | Requested architecture reaches `build_policy`; summary reports the effective model. |
| Training step | Scalar loss supports backward/optimizer updates and leaves targets unchanged. |
| Validation | Evaluation mode and example-weighted aggregation remain correct; stochastic loss uses its own generator. |
| Sampling | Output has the configured horizon and action dimensions, with finite values after denormalization. |
| Rollouts | Current `sample_actions` arguments remain supported; an MSE-like policy may ignore `num_steps`. |
| Checkpoint load | Configuration reconstructs the family; strict weights and dropout names match; model returns in evaluation mode. |
| Further training | Loaded architecture/horizon win over fresh-model CLI settings; optimizer and counters start fresh. |

If a new method needs a different inference control, carry it deliberately through configuration, saving/loading, and evaluation rather than overloading an existing field's meaning.

## 5. Verify behavior, not just shapes

Run relevant existing suites:

```bash
uv run --no-sync pytest \
  tests/test_checkpoint.py tests/test_numerical_correctness.py \
  tests/test_randomness.py tests/test_dropout_integration.py \
  tests/test_evaluation_budgets.py -v
```

Current parameterizations cover the existing families. Passing them alone does not exercise a new family. Update relevant cases for an authorized policy change and agree on any additional coverage beyond that task's scope.

Review these acceptance checks:

1. Initialization is reproducible and leaves default RNG state unchanged.
2. Loss and gradients behave correctly on a small synthetic batch.
3. Sampling returns normalized `(B, H, A)` values and consumes only intended RNG draws.
4. Evaluation mode changes only intended behavior, such as disabling dropout.
5. Saving/loading preserves raw-space predictions under matching inference settings.
6. Fresh and checkpoint-initialized trainer paths use the actual architecture consistently.

Use the [full CI checks](../tests-and-checks.md) after focused checks. A real-data training run is a separate integration step; tests with synthetic data do not establish policy quality.

## Common mistakes

- Adding a class but omitting the type literal or construction dispatch.
- Returning flattened actions where the evaluator expects an action chunk.
- Applying normalization twice or changing coordinate units silently.
- Using global noise or ordinary dropout that bypasses explicit generators.
- Reusing old checkpoint keys after changing their semantics without a compatibility decision.
- Comparing a new loss scale directly with another family's loss instead of comparing rollout behavior.
