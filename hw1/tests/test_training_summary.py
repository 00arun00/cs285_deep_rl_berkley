"""Rendered startup summaries describe the effective run, not ignored inputs."""

from io import StringIO
from pathlib import Path
import unittest

from rich.console import Console

from hw1_imitation.model import PolicyConfig, build_policy
from hw1_imitation.train import TrainConfig, build_training_summary, parse_train_config


class TrainingSummaryTests(unittest.TestCase):
    def summary(self, config=None, policy_type="mse"):
        return dict(
            config=config or TrainConfig(),
            model=build_policy(
                PolicyConfig(
                    policy_type=policy_type,
                    state_dim=5,
                    action_dim=2,
                    chunk_size=8,
                    hidden_dims=(256, 256, 256),
                )
            ),
            run_name="seed_42_[blue]literal[/blue]",
            device="cpu",
            output_dir=Path("/tmp/exp/[blue]literal[/blue]"),
            dataset_path=Path("/tmp/data/pusht/pusht_cchi_v7_replay.zarr"),
            train_episodes=165,
            validation_episodes=41,
            train_samples=20480,
            validation_samples=5170,
            steps_per_epoch=160,
            num_eval_episodes=100,
        )

    def render(self, summary, width=100):
        output = StringIO()
        Console(file=output, width=width, force_terminal=False).print(
            build_training_summary(**summary)
        )
        return output.getvalue()

    def test_default_summary_and_cli(self):
        self.assertTrue(parse_train_config([]).show_summary)
        self.assertFalse(parse_train_config(["--no-show-summary"]).show_summary)
        rendered = self.render(self.summary())
        for expected in (
            "Push-T · Training summary",
            "MSE · initialized from scratch",
            "137,232 trainable",
            "165 train / 41 validation",
            "20,480",
            "5,170",
            "64,000 optimizer steps",
            "Computed from training episodes",
            "Best 3 + latest",
            "100 episodes",
            "[blue]literal[/blue]",
        ):
            self.assertIn(expected, rendered)
        self.assertNotIn("Flow sampling", rendered)
        self.assertNotIn("\x1b", rendered)

    def test_loaded_flow_uses_actual_architecture(self):
        summary = self.summary(
            config=TrainConfig(
                init_from=Path("/tmp/policy.pt"),
                policy_type="mse",
                hidden_dims=(17,),
                chunk_size=3,
                flow_num_steps=7,
            ),
            policy_type="flow",
        )
        rendered = self.render(summary)
        for expected in (
            "FLOW · loaded weights · fresh optimizer",
            "/tmp/policy.pt",
            "256 → 256 → 256",
            "8 steps",
            "Flow sampling",
            "7 steps",
            "Loaded from checkpoint",
        ):
            self.assertIn(expected, rendered)
        self.assertNotIn("MSE", rendered)
        self.assertNotIn("Computed from training episodes", rendered)

    def test_disabled_outputs_and_video_cap(self):
        summary = self.summary(
            config=TrainConfig(
                log_csv=False,
                log_wandb=False,
                save_checkpoints=False,
                num_video_episodes=0,
            )
        )
        rendered = self.render(summary)
        self.assertEqual(rendered.count("Disabled"), 4)
        self.assertNotIn("cs285-hw1", rendered)
        self.assertNotIn("Best 3", rendered)
        enabled = dict(
            summary,
            config=TrainConfig(
                checkpoint_top_k=0,
                num_video_episodes=150,
            ),
        )
        rendered = self.render(enabled)
        self.assertIn("Latest only", rendered)
        self.assertIn("100 episodes/evaluation", rendered)

    def test_narrow_terminal_wraps_without_losing_values(self):
        summary = self.summary()
        wide = self.render(summary, width=110)
        narrow = self.render(summary, width=50)
        self.assertTrue(all(len(line) <= 50 for line in narrow.splitlines()))

        def content(text):
            return "".join(
                char
                for char in text
                if not char.isspace() and char not in "╭╮╰╯─│├┤┬┴┼"
            )

        self.assertEqual(content(wide), content(narrow))
