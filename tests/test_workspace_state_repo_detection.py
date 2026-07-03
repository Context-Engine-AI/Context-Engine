import importlib

import pytest

pytestmark = pytest.mark.unit


def test_extract_repo_name_uses_workspace_relative_leaf(monkeypatch, tmp_path):
    ws_root = tmp_path / "work"
    ws_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("WORKSPACE_PATH", str(ws_root))

    ws = importlib.import_module("scripts.workspace_state")
    ws = importlib.reload(ws)

    def _no_git(*_args, **_kwargs):
        raise AssertionError("git detection should not run for workspace-root-relative paths")

    monkeypatch.setattr(ws.subprocess, "run", _no_git, raising=True)

    repo_path = ws_root / "Context-Engine"
    assert not repo_path.exists()
    assert ws._extract_repo_name_from_path(str(repo_path)) == "Context-Engine"
