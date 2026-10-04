"""CPU tests for reusable policies and fresh training from saved weights."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from hw1_imitation import train
from hw1_imitation.checkpoint import load_policy, save_policy
from hw1_imitation.data import Episode, EpisodesDataset, Normalizer
from hw1_imitation.evaluation import EvaluationResults
from hw1_imitation.model import PolicyConfig, build_policy


class CheckpointTests(unittest.TestCase):
    def test_cli_parses_policy_initialization_and_preserves_defaults(self):
        local = train.parse_train_config([])
        self.assertEqual(local.data_dir, Path("data"))
        self.assertIsNone(local.init_from)
        self.assertEqual(local.checkpoint_top_k, 3)
        self.assertEqual(
            train.parse_train_config(["--checkpoint-top-k", "0"]).checkpoint_top_k, 0
        )
        with self.assertRaisesRegex(ValueError, "checkpoint_top_k"):
            train.TrainConfig(checkpoint_top_k=-1)

        defaults = train.TrainConfig(data_dir=Path("/vol/data"))
        parsed = train.parse_train_config(
            [
                "--init-from",
                "/tmp/policy.pt",
                "--num-epochs",
                "20",
                "--flow-num-steps",
                "3",
                "--data-split-seed",
                "7",
            ],
            defaults=defaults,
        )
        self.assertEqual(parsed.init_from, Path("/tmp/policy.pt"))
        self.assertEqual(parsed.data_dir, defaults.data_dir)
        self.assertEqual(parsed.num_epochs, 20)
        self.assertEqual(parsed.flow_num_steps, 3)
        self.assertEqual(parsed.data_split_seed, 7)
        self.assertEqual(defaults.num_epochs, 400)

    def test_config_rejects_nonpositive_flow_steps(self):
        for value in (0, -1):
            with self.subTest(flow_num_steps=value):
                with self.assertRaisesRegex(
                    ValueError, "flow_num_steps must be positive"
                ):
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
            self.assertEqual(tensor.device.type, "cpu")
            np.testing.assert_array_equal(
                getattr(restored, name), getattr(normalizer, name)
            )
            tensor.fill_(99)
            np.testing.assert_array_equal(
                getattr(restored, name), getattr(normalizer, name)
            )
            self.assertFalse(
                np.shares_memory(getattr(restored, name), getattr(normalizer, name))
            )

    def test_normalizer_rejects_invalid_statistics(self):
        for name, value in (
            ("state_mean", torch.tensor([float("nan"), 0.0])),
            ("action_mean", torch.tensor([float("inf")])),
            ("state_std", torch.tensor([0.0, 1.0])),
            ("action_std", torch.tensor([-1.0])),
            ("action_std", torch.tensor([float("nan")])),
            ("state_std", torch.ones(3)),
            ("state_mean", torch.ones(1, 2)),
        ):
            with self.subTest(name=name, value=value):
                state = {
                    "state_mean": torch.zeros(2),
                    "state_std": torch.ones(2),
                    "action_mean": torch.zeros(1),
                    "action_std": torch.ones(1),
                }
                state[name] = value
                with self.assertRaises(ValueError):
                    Normalizer.from_state_dict(state)
                with self.assertRaises(ValueError):
                    Normalizer(**{key: tensor.numpy() for key, tensor in state.items()})

    def test_policy_round_trip_preserves_raw_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            for policy_type in ("mse", "flow"):
                with self.subTest(policy_type=policy_type):
                    model = build_policy(
                        PolicyConfig(
                            policy_type=policy_type,
                            state_dim=3,
                            action_dim=2,
                            chunk_size=2,
                            hidden_dims=(8,),
                        )
                    )
                    normalizer = Normalizer(
                        np.array([1.0, 2.0, 3.0], dtype=np.float32),
                        np.array([2.0, 3.0, 4.0], dtype=np.float32),
                        np.array([5.0, 6.0], dtype=np.float32),
                        np.array([7.0, 8.0], dtype=np.float32),
                    )
                    path = Path(directory) / policy_type / "policy.pt"
                    save_policy(path, model, normalizer, flow_num_steps=3)
                    restored, stats, inference = load_policy(path, device="cpu")

                    self.assertEqual(restored.config, model.config)
                    self.assertFalse(restored.training)
                    self.assertTrue(model.training)
                    self.assertEqual(inference, {"flow_num_steps": 3})
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    self.assertNotIn("resume", payload)
                    self.assertNotIn("optimizer_state_dict", payload)
                    self.assertNotIn("random_state", payload)
                    for key, value in model.state_dict().items():
                        self.assertEqual(
                            payload["model_state_dict"][key].device.type, "cpu"
                        )
                        torch.testing.assert_close(
                            restored.state_dict()[key],
                            value,
                            rtol=0,
                            atol=0,
                        )

                    raw = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)

                    def predict(policy, norm):
                        # Flow sampling is stochastic; compare with identical
                        # inference noise, not restored training RNG state.
                        torch.manual_seed(123)
                        with torch.no_grad():
                            actions = policy.sample_actions(
                                torch.from_numpy(norm.normalize_state(raw)),
                                num_steps=inference["flow_num_steps"],
                            )
                        return norm.denormalize_action(actions.numpy())

                    np.testing.assert_array_equal(
                        predict(model, normalizer),
                        predict(restored, stats),
                    )

    def test_loaded_policy_starts_a_fresh_training_run(self):
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

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def run(name, config, expected_model=None, pending_cleanup=None):
                target = root / f"{name}.pt"
                steps = 0

                def step(model, optimizer, state, action_chunk):
                    nonlocal steps
                    if steps == 0:
                        # Both entry paths must start with new optimizer history.
                        self.assertEqual(len(optimizer.state), 0)
                        self.assertEqual(optimizer.param_groups[0]["lr"], config.lr)
                        self.assertEqual(
                            optimizer.param_groups[0]["weight_decay"],
                            config.weight_decay,
                        )
                        if expected_model is not None:
                            self.assertEqual(model.config, expected_model.config)
                            for key, value in expected_model.state_dict().items():
                                torch.testing.assert_close(
                                    model.state_dict()[key],
                                    value,
                                    rtol=0,
                                    atol=0,
                                )
                    self.assertEqual(action_chunk.shape[1], model.chunk_size)
                    loss = real_train_step(model, optimizer, state, action_chunk)
                    self.assertTrue(torch.isfinite(loss).item())
                    self.assertTrue(model.training)
                    steps += 1
                    return loss

                def save(
                    model,
                    step,
                    *,
                    normalizer,
                    flow_num_steps,
                    mean_reward,
                    top_k,
                    checkpoints,
                ):
                    self.assertEqual(mean_reward, 0.5)
                    self.assertEqual(top_k, config.checkpoint_top_k)
                    save_policy(
                        target, model, normalizer, flow_num_steps=flow_num_steps
                    )
                    return pending_cleanup or set()

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
                    patch.object(train, "ExperimentLogger") as logger,
                    patch.object(
                        train,
                        "evaluate_policy",
                        return_value=EvaluationResults(0.5, 1, ()),
                    ) as evaluate,
                    patch.object(
                        train, "log_checkpoint_artifact", side_effect=save
                    ) as save_artifact,
                    patch.object(train, "train_step", side_effect=step),
                    patch.object(train.logging, "warning") as warning,
                ):
                    wandb_init.return_value.__enter__.return_value.offline = False
                    train.run_training(config)

                if pending_cleanup:
                    warning.assert_called_once_with(
                        "Training finished with checkpoint cleanup pending for steps %s. "
                        "Extra local files or W&B artifacts may remain.",
                        sorted(pending_cleanup),
                    )
                else:
                    warning.assert_not_called()

                # Four training episodes, each containing six padded samples.
                expected_steps = config.num_epochs * (24 // config.batch_size)
                self.assertEqual(steps, expected_steps)
                logged = [
                    call.kwargs for call in logger.return_value.log_train.call_args_list
                ]
                expected_log_steps = list(
                    range(config.log_interval, expected_steps + 1, config.log_interval)
                )
                if not expected_log_steps or expected_log_steps[-1] != expected_steps:
                    expected_log_steps.append(expected_steps)
                self.assertEqual(
                    [entry["global_step"] for entry in logged], expected_log_steps
                )
                previous_step = 0
                for entry in logged:
                    self.assertEqual(
                        entry["window_examples_count"],
                        (entry["global_step"] - previous_step) * config.batch_size,
                    )
                    previous_step = entry["global_step"]
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(save_artifact.call_count, 1)
                self.assertEqual(save_artifact.call_args.kwargs["step"], expected_steps)
                restored, normalizer, inference = load_policy(target)
                self.assertEqual(
                    evaluate.call_args.kwargs["chunk_size"], restored.chunk_size
                )
                self.assertEqual(inference["flow_num_steps"], config.flow_num_steps)
                return restored, normalizer, target

            for policy_type in ("mse", "flow"):
                with self.subTest(policy_type=policy_type):
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
                            data_split_seed=9,
                            log_interval=5,
                            validation_interval=100,
                            eval_interval=100,
                            num_video_episodes=0,
                            flow_num_steps=4,
                        ),
                        expected_model=original,
                        pending_cleanup={1},
                    )
                    self.assertEqual(trained.config, original.config)
                    for name, value in original_stats.state_dict().items():
                        torch.testing.assert_close(
                            stats.state_dict()[name], value, rtol=0, atol=0
                        )
                    self.assertTrue(
                        any(
                            not torch.equal(value, trained.state_dict()[key])
                            for key, value in original.state_dict().items()
                        )
                    )

    def test_reject_unknown_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.pt"
            torch.save({"format_version": 999}, path)
            with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
                load_policy(path)


if __name__ == "__main__":
    unittest.main()
