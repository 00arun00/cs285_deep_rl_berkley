"""Periodic and final rollouts receive their respective episode budgets."""

from unittest.mock import patch

import numpy as np
from hw1_imitation import train
from hw1_imitation.data import Episode, EpisodesDataset
from hw1_imitation.evaluation import EvaluationResults


class TestEvaluationBudget:
    def test_final_step_uses_final_budget_exactly_once(self, tmp_path):
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
            batch_size=4,
            hidden_dims=(4,),
            chunk_size=1,
            eval_interval=2,
            eval_episodes=5,
            final_eval_episodes=50,
            num_video_episodes=0,
            save_checkpoints=False,
            log_wandb=False,
            log_csv=False,
            show_summary=False,
        )
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
                return_value=EvaluationResults(0.5, 1, ()),
            ) as evaluate,
        ):
            train.run_training(config)

        # Six training steps: periodic rollouts at 2 and 4; final only at 6.
        assert [
            call.kwargs["num_eval_episodes"] for call in evaluate.call_args_list
        ] == [5, 5, 50]
