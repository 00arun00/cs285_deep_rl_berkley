"""Periodic and final rollouts receive their respective episode budgets."""

from unittest.mock import patch

import numpy as np
import pytest
from hw1_imitation import train
from hw1_imitation.data import Episode, EpisodesDataset
from hw1_imitation.evaluation import EvaluationResults


class TestEvaluationBudget:
    @pytest.mark.parametrize(
        "batch_size,interval,expected",
        [
            (4, 2, [(2, 5), (4, 5), (6, 50)]),
            (4, 4, [(4, 5), (6, 50)]),
            (4, 10, [(6, 50)]),
            (24, 1, [(1, 50)]),
        ],
        ids=["aligned-final", "unaligned-final", "short-run", "one-step"],
    )
    def test_final_step_uses_final_budget_exactly_once(
        self, tmp_path, batch_size, interval, expected
    ):
        """Training uses the final rollout budget once at completion.

        Protects:
            Periodic evaluations cannot duplicate or replace the final evaluation on
            aligned steps.
        Value:
            Avoids wasted rollouts and under-budget final scores for short or
            unaligned training runs.
        Approach:
            Observe real optimizer-step counts and substitute only the expensive
            rollout boundary.
        """
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
        config = train.TrainConfig(
            num_epochs=1,
            batch_size=batch_size,
            hidden_dims=(4,),
            chunk_size=1,
            eval_interval=interval,
            eval_episodes=5,
            final_eval_episodes=50,
            num_video_episodes=0,
            save_checkpoints=False,
            log_wandb=False,
            log_csv=False,
            show_summary=False,
        )
        steps = 0
        evaluations = []
        real_step = train.train_step

        def step(*args, **kwargs):
            nonlocal steps
            result = real_step(*args, **kwargs)
            steps += 1
            return result

        def evaluate(**kwargs):
            evaluations.append((steps, kwargs["num_eval_episodes"]))
            return EvaluationResults(0.5, kwargs["num_eval_episodes"], ())

        root = tmp_path
        with (
            patch.object(train, "LOGDIR_PREFIX", str(root)),
            patch.object(train, "download_pusht", return_value=root),
            patch.object(train, "load_episodes_dataset", return_value=episodes),
            patch.object(
                train.torch.accelerator, "current_accelerator", return_value=None
            ),
            patch.object(
                train,
                "evaluate_policy",
                side_effect=evaluate,
            ),
            patch.object(train, "train_step", side_effect=step),
        ):
            train.run_training(config)

        assert evaluations == expected
