"""CPU tests for reusable policies and fresh training from saved weights."""

from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hw1_imitation import train
from hw1_imitation.checkpoint import load_policy, save_policy
from hw1_imitation.data import Episode, EpisodesDataset, Normalizer
from hw1_imitation.evaluation import EvaluationResults
from hw1_imitation.model import BasePolicy, PolicyConfig, build_policy
from hw1_imitation.randomness import RandomStreamFactory, StreamId
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


def test_cli_preserves_config_defaults():
    assert train.parse_train_config([]) == train.TrainConfig()


class TestCheckpoint:
    def test_cli_parses_policy_initialization(self):
        parsed = train.parse_train_config(
            [
                "--init-from",
                "/tmp/policy.pt",
                "--num-epochs",
                "20",
                "--flow-num-steps",
                "3",
                "--data-split-variation",
                "7",
                "--checkpoint-top-k",
                "0",
            ]
        )
        assert parsed.init_from == Path("/tmp/policy.pt")
        assert parsed.num_epochs == 20
        assert parsed.flow_num_steps == 3
        assert parsed.data_split_variation == 7
        assert parsed.checkpoint_top_k == 0

    def test_cli_preserves_supplied_defaults_without_mutating_them(self):
        defaults = train.TrainConfig(
            data_dir=Path("/vol/data"), num_epochs=17, flow_num_steps=6
        )
        before = asdict(defaults)
        parsed = train.parse_train_config(["--num-epochs", "20"], defaults=defaults)
        assert asdict(parsed) == {**before, "num_epochs": 20}
        assert asdict(defaults) == before

    def test_config_rejects_negative_checkpoint_retention(self):
        with pytest.raises(ValueError, match="checkpoint_top_k"):
            train.TrainConfig(checkpoint_top_k=-1)

    @pytest.mark.parametrize("value", (0, -1))
    def test_config_rejects_nonpositive_flow_steps(self, value):
        with pytest.raises(ValueError, match="flow_num_steps must be positive"):
            train.TrainConfig(flow_num_steps=value)

    def test_normalizer_state_does_not_share_storage(self):
        normalizer = Normalizer(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([5.0], dtype=np.float32),
            np.array([6.0], dtype=np.float32),
        )
        state = normalizer.state_dict()
        restored = Normalizer.from_state_dict(state)
        for name, tensor in state.items():
            assert tensor.device.type == "cpu"
            np.testing.assert_array_equal(
                getattr(restored, name), getattr(normalizer, name)
            )
            tensor.fill_(99)
            np.testing.assert_array_equal(
                getattr(restored, name), getattr(normalizer, name)
            )
            assert not np.shares_memory(
                getattr(restored, name), getattr(normalizer, name)
            )

    @pytest.mark.parametrize("group", ["state", "action"])
    @pytest.mark.parametrize(
        "corruption",
        [
            "nan_mean",
            "infinite_mean",
            "nan_std",
            "infinite_std",
            "zero_std",
            "negative_std",
            "empty",
            "shape",
            "rank",
        ],
    )
    @given(data=st.data(), state_dim=st.integers(1, 6), action_dim=st.integers(1, 6))
    def test_normalizer_rejects_invalid_statistics(
        self, group, corruption, data, state_dim, action_dim
    ):
        stats = {}
        for name, size in (("state", state_dim), ("action", action_dim)):
            stats[f"{name}_mean"] = data.draw(
                arrays(np.float32, size, elements=st.floats(-100, 100, width=32))
            )
            stats[f"{name}_std"] = data.draw(
                arrays(np.float32, size, elements=st.floats(0.125, 100, width=32))
            )
        # Establish validity before corrupting exactly one property.
        Normalizer(**stats)
        mean, std = f"{group}_mean", f"{group}_std"
        index = data.draw(st.integers(0, len(stats[mean]) - 1))
        if corruption == "empty":
            stats[mean] = stats[std] = np.array([], dtype=np.float32)
        elif corruption == "shape":
            stats[std] = np.ones(len(stats[mean]) + 1, dtype=np.float32)
        elif corruption == "rank":
            stats[mean] = stats[mean][None, :]
        else:
            field = mean if corruption.endswith("mean") else std
            value = {
                "nan": float("nan"),
                "infinite": float("inf"),
                "zero": 0.0,
                "negative": -1.0,
            }[corruption.split("_")[0]]
            stats[field][index] = value
        with pytest.raises(ValueError):
            Normalizer(**stats)
        with pytest.raises(ValueError):
            Normalizer.from_state_dict(
                {key: torch.from_numpy(value) for key, value in stats.items()}
            )

    def test_reject_unknown_version(self, tmp_path):
        path = tmp_path / "unknown.pt"
        torch.save({"format_version": 999}, path)
        with pytest.raises(ValueError, match="Unsupported checkpoint"):
            load_policy(path)


@pytest.mark.parametrize("policy_type", ["mse", "flow"])
def test_policy_round_trip_preserves_raw_predictions(policy_type, tmp_path):
    model = build_policy(
        PolicyConfig(
            policy_type=policy_type,
            state_dim=3,
            action_dim=2,
            chunk_size=2,
            hidden_dims=(8,),
        ),
        cpu_generator=RandomStreamFactory(42).torch(
            StreamId.MODEL_INIT,
        ),
    )
    normalizer = Normalizer(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([2.0, 3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0], dtype=np.float32),
        np.array([7.0, 8.0], dtype=np.float32),
    )
    path = tmp_path / policy_type / "policy.pt"
    save_policy(path, model, normalizer, flow_num_steps=3)
    restored, stats, inference = load_policy(path, device="cpu")

    assert restored.config == model.config
    assert not restored.training
    assert model.training
    assert inference == {"flow_num_steps": 3}
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert "resume" not in payload
    assert "optimizer_state_dict" not in payload
    assert "random_state" not in payload
    for key, value in model.state_dict().items():
        assert payload["model_state_dict"][key].device.type == "cpu"
        torch.testing.assert_close(
            restored.state_dict()[key],
            value,
            rtol=0,
            atol=0,
        )

    raw = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    streams = RandomStreamFactory(root_seed=123)

    def predict(policy: BasePolicy, norm: Normalizer) -> np.ndarray:
        """Predict with fresh, reproducible inference noise."""
        # Recreate the same stream for each policy so differences
        # in flow noise cannot obscure checkpoint preservation.
        generator = streams.torch(
            StreamId.EVAL_POLICY,
            variation=0,
            index=0,
        )
        with torch.no_grad():
            actions = policy.sample_actions(
                torch.from_numpy(norm.normalize_state(raw)),
                generator=generator,
                num_steps=inference["flow_num_steps"],
            )
        return norm.denormalize_action(actions.numpy())

    np.testing.assert_array_equal(
        predict(model, normalizer),
        predict(restored, stats),
    )


@pytest.mark.parametrize("policy_type", ["mse", "flow"])
def test_loaded_policy_starts_a_fresh_training_run(policy_type, tmp_path):
    rng = np.random.default_rng(7)
    episodes = EpisodesDataset(
        tuple(
            Episode(
                i,
                rng.normal(size=(6, 3)).astype(np.float32),
                rng.normal(size=(6, 2)).astype(np.float32),
            )
            for i in range(5)
        )
    )
    real_train_step = train.train_step

    root = tmp_path

    def run(name, config, expected_model=None):
        target = root / f"{name}.pt"
        steps = 0

        def step(
            model: BasePolicy,
            optimizer: torch.optim.Optimizer,
            state: torch.Tensor,
            action_chunk: torch.Tensor,
            *,
            generator: torch.Generator,
        ) -> torch.Tensor:
            nonlocal steps
            if steps == 0:
                # Both entry paths must start with new optimizer history.
                assert len(optimizer.state) == 0
                assert optimizer.param_groups[0]["lr"] == config.lr
                assert optimizer.param_groups[0]["weight_decay"] == config.weight_decay
                if expected_model is not None:
                    assert model.config == expected_model.config
                    for key, value in expected_model.state_dict().items():
                        torch.testing.assert_close(
                            model.state_dict()[key],
                            value,
                            rtol=0,
                            atol=0,
                        )
            assert action_chunk.shape[1] == model.chunk_size
            # Driver's generator remains unchanged so this wrapper
            # preserves the production stream's progression.
            loss = real_train_step(
                model,
                optimizer,
                state,
                action_chunk,
                generator=generator,
            )
            assert torch.isfinite(loss).item()
            assert model.training
            steps += 1
            return loss

        def save(
            model,
            step,
            *,
            checkpoint_dir,
            run,
            normalizer,
            flow_num_steps,
            mean_reward,
            top_k,
            checkpoints,
        ):
            assert mean_reward == 0.5
            assert top_k == config.checkpoint_top_k
            save_policy(target, model, normalizer, flow_num_steps=flow_num_steps)
            return set()

        with (
            patch.object(train, "LOGDIR_PREFIX", str(root / name)),
            patch.object(train, "download_pusht", return_value=root),
            patch.object(train, "load_episodes_dataset", return_value=episodes),
            patch.object(
                train.torch.accelerator,
                "current_accelerator",
                return_value=None,
            ),
            patch.object(train.wandb, "init") as wandb_init,
            patch.object(
                train,
                "evaluate_policy",
                return_value=EvaluationResults(0.5, 1, ()),
            ) as evaluate,
            patch.object(
                train, "save_checkpoint_and_retain", side_effect=save
            ) as save_artifact,
            patch.object(train, "train_step", side_effect=step),
        ):
            wandb_init.return_value.__enter__.return_value.offline = False
            train.run_training(config)

        # Four training episodes, each containing six padded samples.
        expected_steps = config.num_epochs * (24 // config.batch_size)
        assert steps == expected_steps
        assert evaluate.call_count == 1
        assert save_artifact.call_count == 1
        assert save_artifact.call_args.kwargs["step"] == expected_steps
        restored, normalizer, inference = load_policy(target)
        assert evaluate.call_args.kwargs["chunk_size"] == restored.chunk_size
        assert inference["flow_num_steps"] == config.flow_num_steps
        return restored, normalizer, target

    original, original_stats, path = run(
        f"{policy_type}-fresh",
        train.TrainConfig(
            policy_type=policy_type,
            hidden_dims=(8,),
            chunk_size=2,
            num_epochs=1,
            batch_size=4,
            lr=3e-4,
            log_interval=4,
            validation_interval=100,
            eval_interval=100,
            num_video_episodes=0,
            flow_num_steps=3,
        ),
    )
    trained, stats, _ = run(
        f"{policy_type}-loaded",
        train.TrainConfig(
            init_from=path,
            # These new-model settings must not replace the
            # loaded architecture or its action horizon.
            policy_type="flow" if policy_type == "mse" else "mse",
            hidden_dims=(16, 16),
            chunk_size=5,
            num_epochs=2,
            batch_size=3,
            lr=1e-3,
            weight_decay=0.01,
            seed=8,
            data_split_variation=9,
            log_interval=5,
            validation_interval=100,
            eval_interval=100,
            num_video_episodes=0,
            flow_num_steps=4,
        ),
        expected_model=original,
    )
    assert trained.config == original.config
    for name, value in original_stats.state_dict().items():
        torch.testing.assert_close(stats.state_dict()[name], value, rtol=0, atol=0)
    assert any(
        (
            not torch.equal(value, trained.state_dict()[key])
            for key, value in original.state_dict().items()
        )
    )
