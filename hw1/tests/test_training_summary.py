"""Rendered startup summaries describe the effective run, not ignored inputs."""

from io import StringIO
from pathlib import Path

from rich.console import Console

from hw1_imitation.config import (
    CheckpointConfig,
    EvaluationConfig,
    LoggingConfig,
    OptimizationConfig,
    TrainConfig,
)
from hw1_imitation.model import PolicyConfig, build_policy
from hw1_imitation.randomness import RandomStreamFactory, StreamId
from hw1_imitation.reporting import build_training_summary

from .summary_helpers import summary_row_value


class TestTrainingSummary:
    def summary(self, config=None, policy_type="mse"):
        model = build_policy(
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
        )
        return dict(
            config=config or TrainConfig(),
            architecture=model.config,
            trainable_parameters=sum(
                p.numel() for p in model.parameters() if p.requires_grad
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

    def test_default_summary_reports_effective_values(self):
        """Catches misleading run budgets and accidental markup interpretation in startup
        output."""
        rendered = self.render(
            self.summary(
                config=TrainConfig(
                    optimization=OptimizationConfig(num_epochs=7),
                    evaluation=EvaluationConfig(
                        interval=123, episodes=5, final_episodes=50
                    ),
                )
            )
        )
        expected = {
            "Policy": "MSE",
            "Parameters": "137,232",
            "Episodes": "165 train / 41 validation",
            "Training samples": "20,480",
            "Validation samples": "5,170",
            "Training budget": "1,120 optimizer steps",
            "Normalization": "training episodes",
            "Checkpoints": "Best 3",
            "Final evaluation": "50 episodes",
        }
        for label, value in expected.items():
            assert value in summary_row_value(rendered, label)
        rollout = summary_row_value(rendered, "Rollout evaluation")
        assert "123 steps" in rollout
        assert "5 episodes" in rollout
        assert "[blue]literal[/blue]" in rendered
        assert "Flow sampling" not in rendered
        assert "\x1b" not in rendered

    def test_loaded_flow_uses_actual_architecture(self):
        """Prevents ignored new-model flags from misrepresenting a loaded policy in the
        summary."""
        summary = self.summary(
            config=TrainConfig(
                warm_start=Path("/tmp/policy.pt"),
                model=None,
                evaluation=EvaluationConfig(flow_num_steps=7),
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
        """Keeps the displayed output settings consistent with the requested storage and
        video budget."""
        summary = self.summary(
            config=TrainConfig(
                logging=LoggingConfig(csv=False, wandb=False),
                checkpoints=CheckpointConfig(enabled=False),
                evaluation=EvaluationConfig(num_video_episodes=0),
            )
        )
        rendered = self.render(summary)
        for label in ("CSV logging", "W&B", "Checkpoints", "Videos"):
            assert summary_row_value(rendered, label) == "Disabled"
        assert "cs285-hw1" not in rendered
        assert "Best 3" not in rendered
        enabled = dict(
            summary,
            config=TrainConfig(
                checkpoints=CheckpointConfig(top_k=0),
                evaluation=EvaluationConfig(num_video_episodes=150),
            ),
        )
        rendered = self.render(enabled)
        assert "Latest only" in summary_row_value(rendered, "Checkpoints")
        assert "150 episodes" in summary_row_value(rendered, "Videos")

    def test_narrow_terminal_wraps_without_losing_values(self):
        """Catches clipping of run information on narrow terminals without fixing
        whitespace layout."""
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
