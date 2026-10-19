"""Individual runs and W&B sweeps share explicit-value Tyro syntax."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from wandb.sdk.launch.sweeps.utils import create_sweep_command_args

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


@pytest.mark.parametrize("policy", ["mse", "flow"])
def test_yaml_parameters_survive_wandb_argument_expansion(policy):
    path = Path(__file__).parents[1] / "conf" / "sweeps" / f"{policy}.yaml"
    sweep = yaml.safe_load(path.read_text())
    values = {}
    for name, spec in sweep["parameters"].items():
        if "value" in spec:
            value = spec["value"]
        elif "values" in spec:
            value = spec["values"][0]
        else:
            value = spec["min"]
        values[name] = {"value": value}

    assert sweep["command"] == [
        "${env}",
        "uv",
        "run",
        "python",
        "-m",
        "hw1_imitation.train",
        "${args}",
    ]
    macros = create_sweep_command_args({"args": values})
    with (
        patch("sys.argv", ["train", *macros["args"]]),
        patch.object(train, "run_training") as run,
    ):
        train.main()

    run.assert_called_once()
    config = run.call_args.args[0]
    expected = {name: spec["value"] for name, spec in values.items()}
    expected["hidden_dims"] = (128, 128, 128)
    assert config == TrainConfig(**expected)
    assert config.policy_type == policy
    assert config.pad_action_chunk_with_last_action is True
    assert config.early_stopping_min_delta == 0.001
