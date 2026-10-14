"""Keep training's temporary checkpoints inside each test's sandbox."""

import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolate_checkpoint_tempdir(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("checkpoint-temp")
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root
