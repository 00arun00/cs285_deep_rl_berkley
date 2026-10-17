# Push-T: task, demonstrations, and policy inputs

Push-T is a planar manipulation task: a circular pusher moves a T-shaped block toward a target pose. The action is the pusher's target `(x, y)` position, not a force or a displacement. The environment bounds each coordinate to `[0, 512]`. [Environment description](https://github.com/huggingface/gym-pusht#description).

The implementation in `hw1/` learns from demonstrations, then measures behavior in simulated rollouts. It does not update the policy from rollout rewards. For commands, start with the [orientation](../index.md).

## Separate the task from its representations

| Representation | Information given to a policy | Used here? |
| --- | --- | --- |
| State | Pusher position, block position, and block angle: five numbers. | Yes. |
| Keypoints | Positions describing the block's geometry, with pusher position supplied separately in the Gym package. | No. |
| Pixels | An RGB image; a separate variant also includes pusher position. | No. |

These are observation choices for the same task, not interchangeable tensor formats. The Gym package calls its keypoint-style option `environment_state_agent_pos`; image options are `pixels` and `pixels_agent_pos`. [Supported observations](https://github.com/huggingface/gym-pusht#observation-space).

The state vector is ordered `[pusher_x, pusher_y, block_x, block_y, block_angle]`. Positions and action targets use the same simulation coordinate frame, bounded to `[0, 512]`; these are environment coordinates, not physical meters. The angle is in radians. [State/action implementation](https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/envs/pusht.py).

The local evaluator explicitly creates:

```python
gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
```

`render_mode="rgb_array"` enables frames for optional videos. It does **not** make images the policy's observations. The locked `gym-pusht` version is 0.1.6; its registration limits an episode to 300 steps. [Upstream registration](https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/__init__.py).

## Where the demonstrations come from

The downloader uses the simulation archive distributed with **Diffusion Policy**. That project documents `pusht.zip` and the Zarr replay layout; `gym-pusht` is an adaptation of its environment. These provenance links do not mean this repository implements Diffusion Policy's diffusion model. [Training-data distribution](https://github.com/real-stanford/diffusion_policy#download-training-data), [environment acknowledgement](https://github.com/huggingface/gym-pusht#acknowledgment).

The upstream collection script records mouse-controlled demonstrations, pairing the current state with the commanded action and also recording images and keypoints. This explains the multiple fields in the replay store; the downloaded artifact was not inspected or recollected while writing this guide. [Collection source](https://github.com/real-stanford/diffusion_policy/blob/main/demo_pusht.py).

## Which artifact and fields this code consumes

| Item | Local contract |
| --- | --- |
| Download URL | `https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip` |
| Extracted path | `<data-dir>/pusht/pusht_cchi_v7_replay.zarr` |
| `data/state` | Expected `(N, 5)` state rows, converted to float32. |
| `data/action` | Expected `(N, 2)` action rows, converted to float32. |
| `meta/episode_ends` | Expected `(E,)` exclusive episode-end offsets, converted to int64. |

The loader reads only these three arrays. Other archive fields, such as images or keypoints, are not model inputs. ZIP is the distribution container; Zarr is the array store inside it. A repackaged dataset with the same task name is not automatically compatible with these paths or semantics. [Upstream replay layout](https://github.com/real-stanford/diffusion_policy#replaybuffer).

Here `N` is the number of recorded timesteps and `E` the number of episodes; these symbols do not assert measured counts. These shapes describe the expected state-based artifact interface. The code derives feature dimensions from loaded data rather than enforcing five/two features or a fixed demonstration count. Do not treat comments containing counts as validation of a local dataset. The [data pipeline](data-pipeline.md) explains episode and shape checks.

## Keep compatibility questions separate

The `v7` in `pusht_cchi_v7_replay.zarr` identifies this replay artifact; it does not identify every dataset or environment called Push-T.

| What changes? | What to verify before comparing or reusing it |
| --- | --- |
| Dataset release | The demonstrations themselves, including trajectory contents and episode membership, may differ. |
| Packaging conversion | Container or field names may change; verify state/action ordering, coordinate scales, alignment, and exclusive episode offsets rather than assuming a format conversion preserves them. |
| Observation representation | State, keypoints, and images require different model inputs, even if recorded from the same trajectories. |
| Environment or evaluation setup | Version, reset distribution, time limit, reward definition, and action execution horizon can affect scores independently of the dataset. |

For this implementation, compatibility means matching the consumed fields and their semantics **and** evaluating with the intended environment settings. ZIP-versus-Zarr alone says nothing about those contracts.

## Two policy families, one interface

Both policies take one normalized state and return a normalized action chunk shaped `(batch, chunk_size, action_dim)`.

| Policy | Training target | Inference |
| --- | --- | --- |
| `mse` | Demonstrated action chunk; minimize mean squared error. | Predict the chunk directly. |
| `flow` | Velocity from Gaussian noise toward the demonstrated chunk, at a sampled interpolation time. | Integrate the learned velocity field from fresh Gaussian noise using forward Euler steps. |

The default fresh model is MSE with an eight-action chunk. Flow sampling defaults to ten integration steps. Both use the same demonstration pipeline and rollout evaluator; their loss values describe different objectives and should not be compared as a shared quality score.

Continue with [demonstrations to batches](data-pipeline.md), [evaluation](evaluation.md), or the [policy reference](../../reference/policies.md).
