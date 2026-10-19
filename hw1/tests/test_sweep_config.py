"""Individual runs and W&B sweeps share explicit-value Tyro syntax."""

from pathlib import Path
from unittest.mock import patch

import pytest

from hw1_imitation import train
from hw1_imitation.train import TrainConfig, parse_train_config


@pytest.mark.parametrize("padding", [True, False])
def test_sweep_entry_point_passes_typed_config_to_shared_trainer(padding):
    args = [
        "train",
        "--policy_type=flow",
        "--hidden_dims=(128, 256, 512)",
        f"--pad_action_chunk_with_last_action={padding}",
        "--lr=0.001",
        "--data_dir=data with spaces",
        "--log_wandb=True",
    ]
    with patch("sys.argv", args), patch.object(train, "run_training") as run:
        train.main()
    run.assert_called_once()
    config = run.call_args.args[0]
    assert config.policy_type == "flow"
    assert config.hidden_dims == (128, 256, 512)
    assert config.pad_action_chunk_with_last_action is padding
    assert config.lr == 0.001
    assert config.data_dir == Path("data with spaces")
    assert config.log_wandb is True
    assert config.batch_size == TrainConfig().batch_size


def test_individual_training_cli_uses_explicit_values_and_caller_defaults():
    defaults = TrainConfig(data_dir=Path("/vol/data"))
    config = parse_train_config(
        ["--log-wandb", "False", "--hidden-dims", "(128, 256, 512)"],
        defaults=defaults,
    )
    assert config.log_wandb is False
    assert config.hidden_dims == (128, 256, 512)
    assert config.data_dir == defaults.data_dir
    assert parse_train_config([]) == TrainConfig()
