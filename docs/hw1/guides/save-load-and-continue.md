# Save, load, and train further

Use a policy checkpoint to run inference, evaluate a saved policy, or initialize another training run. Complete [setup](../setup.md) and work from **`hw1/`**. Loading for inference does not need the original demonstration dataset.

## Save the model and its normalization together

The trainer saves at every selection evaluation. Persistent files use `exp/<run>/checkpoints/policy_step_<step>.pt`; see [Inspect results](inspect-results.md) to choose one.

When working directly with a policy and its matching normalizer, use the existing API:

```python
from hw1_imitation.checkpoint import save_policy

# model and normalizer are an existing, matching pair.
save_policy("policy.pt", model, normalizer, flow_num_steps=10)
```

This writes a file and replaces an existing destination only after serialization succeeds. Keep model state stable during saving; do not have multiple writers target the same path. A policy file stores weights, architecture, normalization, dropout settings, and inference defaults. See [the checkpoint contract](../concepts/checkpoints.md) for omissions and failure behavior.

## Load for inference

Provide an existing checkpoint and a NumPy batch of raw observations shaped `(batch, model.state_dim)`. For this Push-T state interface, each observation contains the ordered five state values described in [Push-T](../concepts/pusht.md).

```python
import torch
from hw1_imitation.checkpoint import load_policy
from hw1_imitation.randomness import RandomStreamFactory, StreamId

model, normalizer, inference = load_policy("policy.pt", device="cpu")
# raw_observations must be supplied from your compatible environment/data.
states = torch.as_tensor(
    normalizer.normalize_state(raw_observations), dtype=torch.float32,
)
streams = RandomStreamFactory(root_seed=42)
generator = streams.torch(StreamId.EVAL_POLICY, index=0, device=states.device)
with torch.no_grad():
    predicted = model.sample_actions(
        states, generator=generator, num_steps=inference["flow_num_steps"],
    )
actions = normalizer.denormalize_action(predicted.cpu().numpy())
```

Expected `actions` shape is `(batch, model.chunk_size, model.action_dim)` in raw action coordinates. The loader returns the model in evaluation mode. Clip to the environment's action bounds before stepping it; manual sampling above does not perform clipping.

Retain the generator across sequential predictions if you want its draws to advance. Recreating it with the same identity restarts the noise stream. For accelerator inference, move both the input tensor and generator to the model's device.

## Evaluate in the supported environment

For a compatible saved Push-T policy, this API call runs real environment episodes but does not train or download demonstrations:

```python
from hw1_imitation.evaluation import evaluate_policy

result = evaluate_policy(
    model=model,
    normalizer=normalizer,
    device=next(model.parameters()).device,
    chunk_size=model.chunk_size,
    video_size=(256, 256),
    num_video_episodes=0,
    num_eval_episodes=10,
    flow_num_steps=inference["flow_num_steps"],
    streams=streams,
)
print(result.mean_reward, result.num_episodes)
```

This example assumes the model, normalizer, inference settings, and factory from the preceding snippet. It uses the evaluator's default selection streams. Ten episodes is an illustrative budget; record the budget and stream choices when comparing results. The evaluator handles normalization, action clipping, episode boundaries, and model-mode restoration.

## Initialize another training run

Replace the checkpoint path before running this real training command:

```bash
uv run --no-sync src/hw1_imitation/train.py \
  --init-from /path/to/policy.pt \
  --num-epochs 20 --lr 0.0001 \
  --log-wandb False --num-video-episodes 0 \
  --exp-name continued-policy
```

This starts a new 20-epoch run. It may download demonstrations and writes a new experiment directory, CSVs, and retained checkpoints. The command was source-checked, not executed for this guide.

| State | Behavior with `--init-from` |
| --- | --- |
| Architecture, weights, normalizer | Restored from the file. Fresh-model flags `--policy-type`, `--hidden-dims`, and `--chunk-size` do not replace them. |
| Dropout probabilities | Inherited per layer when `--dropout-p` is omitted; an explicit value overrides all layers. Use `--dropout-p 0` to disable them. |
| Flow evaluation steps | Taken from the new invocation's `--flow-num-steps` or its default, not inherited by the trainer from the checkpoint. |
| Optimizer, counters, root seed and generators | Created fresh from the new invocation; no saved optimizer or RNG position exists. |
| Data split, epoch budget, learning rate, monitoring | Taken from the new invocation and its defaults. |
| Logs and retention history | Start in the new run. |

A “fresh seed” means newly initialized streams from the selected seed, which may have the same integer value as before. It does not mean the trainer chooses a new random seed automatically.

## Verify what was restored

1. Confirm the startup summary reports loaded weights and the intended checkpoint.
2. Check the effective architecture and horizon, especially if fresh-model flags were also supplied.
3. Verify that new data has the same feature meanings, dimensions, and units; the saved normalizer is reused without refitting.
4. Inspect the new run's outputs and final checkpoint selection using [Inspect results](inspect-results.md).

Do not call this exact interrupted-run resumption: optimizer moments, RNG positions, and training position are absent. Do not bypass unsupported-version errors by loading legacy whole-model pickle files through another path. For deliberate format work, use [Change the checkpoint format](change-checkpoint-format.md).
