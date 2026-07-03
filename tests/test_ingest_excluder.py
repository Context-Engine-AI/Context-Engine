import os
import importlib

ing = importlib.import_module("scripts.ingest_code")


def test_excluder_defaults_and_ignore(tmp_path, monkeypatch):
    root = tmp_path
    # Create files and dirs
    (root / ".git").mkdir()
    (root / "node_modules").mkdir()
    (root / ".remote-git").mkdir()
    (root / "keep.py").write_text("print('ok')\n")
    (root / "skip.log").write_text("log\n")

    # .qdrantignore excludes *.log and folder tmp/
    (root / ".qdrantignore").write_text("""\n*.log\ntmp/\n""")

    excl = ing._Excluder(root)
    # Defaults should exclude .git and node_modules directories
    assert excl.exclude_dir("/.git")
    assert excl.exclude_dir("/node_modules")
    assert excl.exclude_dir("/.remote-git")
    # .qdrantignore should exclude .log files
    assert excl.exclude_file("skip.log")
    # keep.py should be allowed
    assert not excl.exclude_file("keep.py")


def test_excluder_env_overrides(tmp_path, monkeypatch):
    root = tmp_path
    (root / "a").mkdir()
    (root / "a" / "file.tmp").write_text("x\n")

    # Disable defaults and add custom patterns via QDRANT_EXCLUDES (comma-separated)
    monkeypatch.setenv("QDRANT_DEFAULT_EXCLUDES", "0")
    monkeypatch.setenv("QDRANT_EXCLUDES", "build,dist,*.tmp,*.bak")

    excl = ing._Excluder(root)
    assert not excl.exclude_dir("/.git")  # defaults off
    assert excl.exclude_file("a/file.tmp")


def test_remote_git_history_manifest_detection(tmp_path):
    p = tmp_path / ".remote-git" / "git_history_test.json"
    assert ing._should_skip_explicit_file_by_excluder(p)


def test_iter_files_single_file_skips_excluded_remote_git_manifest(tmp_path):
    p = tmp_path / ".remote-git" / "git_history_test.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}\n")

    assert ing._should_skip_explicit_file_by_excluder(p)
    assert list(ing.iter_files(p)) == []


def test_excluder_default_excludes_remote_git_nested(tmp_path):
    root = tmp_path
    repo = root / "repo"
    (repo / ".remote-git").mkdir(parents=True, exist_ok=True)

    excl = ing._Excluder(root)
    assert excl.exclude_dir("/repo/.remote-git")
