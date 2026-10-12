"""Sweep trials rank retained policies using one separate final evaluation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

from hw1_imitation import train, evaluation
from hw1_imitation.data import Episode, EpisodesDataset
from hw1_imitation.evaluation import EvaluationResults
from hw1_imitation.randomness import RandomStreamFactory, StreamId
from tests.test_randomness import TinyEnv, make_policy, make_normalizer


@pytest.mark.parametrize(
    "early_stop,interval,best_step,last_step",
    [(False, 1, 2, 6), (True, 1, 2, 3), (False, 4, 4, 6), (False, 100, 6, 6)],
)
@pytest.mark.parametrize("online", [False, True])
def test_final_evaluation_restores_best_and_reports_once(
    tmp_path, early_stop, interval, best_step, last_step, online
):
    episodes = EpisodesDataset(
        tuple(
            Episode(i, np.zeros((6, 3), np.float32), np.zeros((6, 2), np.float32))
            for i in range(5)
        )
    )
    config = train.TrainConfig(
        hidden_dims=(4,),
        chunk_size=1,
        batch_size=24,
        num_epochs=6,
        eval_interval=interval,
        eval_episodes=2,
        final_eval_episodes=7,
        num_video_episodes=0,
        checkpoint_top_k=1,
        log_wandb=online,
        log_csv=False,
        show_summary=False,
        final_eval_use_best_checkpoint=True,
        early_stopping_patience=2 if early_stop else None,
        early_stopping_min_epochs=1,
        early_stopping_min_delta=0.125,
    )
    step = 0
    observed = []
    scores = [0.5, 0.5625, 0.4, 0.5625, 0.3, 0.2]

    def train_step(*, model, **kwargs):
        nonlocal step
        step += 1
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(step)
        return torch.tensor(float(step))

    def evaluate(**kwargs):
        final = kwargs.get("final_evaluation", False)
        observed.append((step, kwargs["num_eval_episodes"], final))
        if final:
            # Step 2 has the best reward, despite an improvement below min_delta.
            # Step 4 ties it in the full-length run and must not replace it.
            assert all(torch.all(p == best_step) for p in kwargs["model"].parameters())
            assert not kwargs["model"].training
            assert kwargs["num_video_episodes"] == 0
            score = 0.9
        else:
            score = scores[step - 1]
        return EvaluationResults(score, kwargs["num_eval_episodes"], ())

    run = MagicMock()
    run.offline = False
    run.summary = {}
    run.__enter__.return_value = run
    real_save = train.save_checkpoint_and_retain

    def save_without_upload(**kwargs):
        # Exercise real disk saving and pruning without remote artifacts.
        return real_save(**{**kwargs, "run": None})

    with (
        patch.object(train, "LOGDIR_PREFIX", str(tmp_path)),
        patch.object(train, "download_pusht", return_value=tmp_path),
        patch.object(train, "load_episodes_dataset", return_value=episodes),
        patch.object(train.torch.accelerator, "current_accelerator", return_value=None),
        patch.object(train, "train_step", side_effect=train_step),
        patch.object(train, "evaluate_policy", side_effect=evaluate),
        patch.object(
            train, "save_checkpoint_and_retain", side_effect=save_without_upload
        ),
        patch.object(train.wandb, "init", return_value=run) as init,
    ):
        train.run_training(config)

    assert step == last_step
    assert observed == [
        (i, 2, False)
        for i in range(1, last_step + 1)
        if i % interval == 0 or i == last_step
    ] + [(last_step, 7, True)]
    files = list(tmp_path.glob("*/final_eval.json"))
    assert len(files) == 1
    result = json.loads(files[0].read_text())
    assert result == {
        "global_step": last_step,
        "final/mean_reward": 0.9,
        "final/num_episodes": 7,
        "final/checkpoint_step": best_step,
        "final/selection_mean_reward": scores[best_step - 1],
        "stop_reason": "plateau" if early_stop else "max_epochs",
    }
    assert {p.name for p in files[0].parent.glob("checkpoints/*.pt")} == {
        f"policy_step_{best_step}.pt",
        f"policy_step_{last_step}.pt",
    }
    if online:
        final_logs = [
            call.args[0]
            for call in run.log.call_args_list
            if "final/mean_reward" in call.args[0]
        ]
        assert len(final_logs) == 1
        assert final_logs[0]["final/checkpoint_step"] == best_step
        assert run.summary["stop_reason"] == result["stop_reason"]
        assert init.call_args.kwargs["config"]["batch_size"] == 24
    else:
        init.assert_not_called()


@pytest.mark.parametrize(
    "settings",
    [
        {"save_checkpoints": False},
        {"checkpoint_top_k": 0},
    ],
)
def test_best_final_evaluation_requires_retained_checkpoints(settings):
    with pytest.raises(ValueError, match="Best-checkpoint"):
        train.TrainConfig(final_eval_use_best_checkpoint=True, **settings)


def test_final_streams_are_separate_and_replayable():
    policy = make_policy()
    streams = RandomStreamFactory(42)

    def rollout(final):
        env = TinyEnv()
        with patch.object(evaluation.gym, "make", return_value=env):
            evaluation.evaluate_policy(
                model=policy,
                normalizer=make_normalizer(),
                device=torch.device("cpu"),
                chunk_size=1,
                video_size=(256, 256),
                num_video_episodes=0,
                flow_num_steps=2,
                streams=streams,
                num_eval_episodes=2,
                final_evaluation=final,
                show_progress=False,
            )
        return np.array(env.initial_observations), np.array(env.actions)

    periodic_obs, _ = rollout(False)
    final_obs, final_actions = rollout(True)
    replay_obs, replay_actions = rollout(True)
    assert not np.array_equal(periodic_obs, final_obs)
    np.testing.assert_array_equal(final_obs, replay_obs)
    np.testing.assert_array_equal(final_actions, replay_actions)
    assert streams.seed(StreamId.EVAL_POLICY) != streams.seed(
        StreamId.FINAL_EVAL_POLICY
    )


@pytest.mark.parametrize("policy", ["mse", "flow"])
def test_yaml_parameters_survive_wandb_argument_expansion(policy):
    # Use the installed W&B macro expansion, not a hand-built command approximation.
    from wandb.sdk.launch.sweeps.utils import create_sweep_command_args

    path = Path(__file__).parents[1] / "sweeps" / f"{policy}.yaml"
    sweep = yaml.safe_load(path.read_text())
    values = {}
    for name, spec in sweep["parameters"].items():
        value = (
            spec["value"]
            if "value" in spec
            else spec["values"][0]
            if "values" in spec
            else spec["min"]
        )
        values[name] = {"value": value}
    macros = create_sweep_command_args({"args": values})
    assert sweep["command"] == [
        "${env}",
        "uv",
        "run",
        "python",
        "-m",
        "hw1_imitation.train",
        "${args}",
    ]
    with (
        patch("sys.argv", ["train", *macros["args"]]),
        patch.object(train, "run_training") as run,
    ):
        train.main()
    config = run.call_args.args[0]
    assert config.policy_type == policy
    assert config.hidden_dims == (128, 128, 128)
    assert config.final_eval_use_best_checkpoint
    assert config.eval_episodes == 100
    assert config.validation_interval == 1000
    assert config.early_stopping_patience == 7
    assert config.early_stopping_min_epochs == 50
    assert config.early_stopping_min_delta == (0.01 if policy == "mse" else 0.001)


@pytest.mark.parametrize("policy", ["mse", "flow"])
def test_yaml_objective_is_best_periodic_eval_reward(policy):
    """Sweeps rank runs by the metric's summary; the objective must be eval/mean_reward."""
    path = Path(__file__).parents[1] / "sweeps" / f"{policy}.yaml"
    sweep = yaml.safe_load(path.read_text())
    assert sweep["metric"] == {"name": "eval/mean_reward", "goal": "maximize"}
