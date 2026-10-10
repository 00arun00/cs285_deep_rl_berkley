"""Rendered startup summaries describe the effective run, not ignored inputs."""

import re
from io import StringIO
from pathlib import Path

from hw1_imitation.model import PolicyConfig, build_policy
from hw1_imitation.randomness import RandomStreamFactory, StreamId
from hw1_imitation.train import TrainConfig, build_training_summary, parse_train_config
from rich.console import Console


class TestTrainingSummary:
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
                ),
                cpu_generator=RandomStreamFactory(42).torch(
                    StreamId.MODEL_INIT,
                ),
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
        )

    def render(self, summary, width=100):
        output = StringIO()
        Console(file=output, width=width, force_terminal=False).print(
            build_training_summary(**summary)
        )
        return output.getvalue()

    def test_default_summary_and_cli(self):
        assert parse_train_config([]).show_summary
        assert not parse_train_config(["--no-show-summary"]).show_summary
        rendered = self.render(
            self.summary(config=TrainConfig(eval_episodes=5, final_eval_episodes=50))
        )
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
            "[blue]literal[/blue]",
        ):
            assert expected in rendered
        assert re.search(
            "Rollout evaluation\\s*│\\s*Every 10,000 steps · 5 episodes", rendered
        )
        assert re.search("Final evaluation\\s*│\\s*50 episodes", rendered)
        assert "Flow sampling" not in rendered
        assert "\x1b" not in rendered

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
            assert expected in rendered
        assert "MSE" not in rendered
        assert "Computed from training episodes" not in rendered

    def test_disabled_outputs_and_configured_video_count(self):
        summary = self.summary(
            config=TrainConfig(
                log_csv=False,
                log_wandb=False,
                save_checkpoints=False,
                num_video_episodes=0,
            )
        )
        rendered = self.render(summary)
        assert rendered.count("Disabled") == 4
        assert "cs285-hw1" not in rendered
        assert "Best 3" not in rendered
        enabled = dict(
            summary,
            config=TrainConfig(
                checkpoint_top_k=0,
                num_video_episodes=150,
            ),
        )
        rendered = self.render(enabled)
        assert "Latest only" in rendered
        assert "Up to 150 episodes" in rendered

    def test_narrow_terminal_wraps_without_losing_values(self):
        summary = self.summary()
        wide = self.render(summary, width=110)
        narrow = self.render(summary, width=50)
        assert all((len(line) <= 50 for line in narrow.splitlines()))

        def content(text):
            return "".join(
                char
                for char in text
                if not char.isspace() and char not in "╭╮╰╯─│├┤┬┴┼"
            )

        assert content(wide) == content(narrow)
