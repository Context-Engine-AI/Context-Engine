#!/usr/bin/env python3
"""
Comprehensive tests for workspace_state.py.

Tests cover:
- State dataclass/TypedDict structures
- Collection name generation and sanitization
- State file operations (read/write/atomic updates)
- Multi-repo mode handling
- File locking mechanisms
- Environment variable gating
"""
import importlib
import json
import os
import tempfile
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# Fixture: Isolated workspace_state import
# ============================================================================
@pytest.fixture
def ws_module(monkeypatch, tmp_path):
    """
    Import workspace_state with isolated environment and temp workspace.
    Returns the reloaded module.
    """
    ws_root = tmp_path / "work"
    ws_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("WORKSPACE_PATH", str(ws_root))
    monkeypatch.setenv("WATCH_ROOT", str(ws_root))
    monkeypatch.delenv("MULTI_REPO_MODE", raising=False)
    monkeypatch.delenv("LOGICAL_REPO_REUSE", raising=False)

    ws = importlib.import_module("scripts.workspace_state")
    ws = importlib.reload(ws)
    return ws


# ============================================================================
# Tests: Sanitize Name
# ============================================================================
class TestSanitizeName:
    """Tests for _sanitize_name function."""

    def test_sanitize_basic(self, ws_module):
        """Basic name sanitization."""
        assert ws_module._sanitize_name("My-Project") == "my-project"
        assert ws_module._sanitize_name("  UPPERCASE  ") == "uppercase"

    def test_sanitize_special_chars(self, ws_module):
        """Special characters are converted to hyphens."""
        assert ws_module._sanitize_name("foo@bar!baz") == "foo-bar-baz"
        assert ws_module._sanitize_name("a/b\\c:d") == "a-b-c-d"

    def test_sanitize_consecutive_hyphens(self, ws_module):
        """Multiple consecutive hyphens are collapsed."""
        assert ws_module._sanitize_name("foo---bar") == "foo-bar"
        assert ws_module._sanitize_name("a@@@b") == "a-b"

    def test_sanitize_empty_string(self, ws_module):
        """Empty string falls back to 'workspace'."""
        assert ws_module._sanitize_name("") == "workspace"
        assert ws_module._sanitize_name("   ") == "workspace"
        assert ws_module._sanitize_name("@#$%") == "workspace"

    def test_sanitize_max_length(self, ws_module):
        """Truncates to max_len."""
        long_name = "a" * 100
        result = ws_module._sanitize_name(long_name, max_len=10)
        assert len(result) == 10


# ============================================================================
# Tests: Collection Name Generation
# ============================================================================
class TestCollectionNameGeneration:
    """Tests for _generate_collection_name function."""

    def test_generates_consistent_hash(self, ws_module, tmp_path):
        """Same path generates same collection name."""
        ws_path = tmp_path / "my-repo"
        ws_path.mkdir()

        name1 = ws_module._generate_collection_name(str(ws_path))
        name2 = ws_module._generate_collection_name(str(ws_path))

        assert name1 == name2
        assert "my-repo-" in name1  # Contains repo name
        assert len(name1.split("-")[-1]) == 6  # 6 char hash suffix

    def test_different_paths_different_hashes(self, ws_module, tmp_path):
        """Different paths get different hashes."""
        path1 = tmp_path / "repo-a"
        path2 = tmp_path / "repo-b"
        path1.mkdir()
        path2.mkdir()

        name1 = ws_module._generate_collection_name(str(path1))
        name2 = ws_module._generate_collection_name(str(path2))

        assert name1 != name2


# ============================================================================
# Tests: Collection Name Resolution
# ============================================================================
class TestCollectionNameResolution:
    """Tests for get_collection_name() precedence rules."""

    def test_multi_repo_does_not_hard_override_with_env(self, ws_module, monkeypatch):
        """In multi-repo mode, COLLECTION_NAME must not override per-repo naming."""
        monkeypatch.setenv("MULTI_REPO_MODE", "1")
        monkeypatch.setenv("COLLECTION_NAME", "codebase")
        ws = importlib.reload(ws_module)

        coll = ws.get_collection_name("my-repo_old")
        assert coll != "codebase_old"
        assert coll != "codebase"
        assert coll.endswith("_old")

    def test_single_repo_env_override_preserved(self, ws_module, monkeypatch):
        """In single-repo mode, COLLECTION_NAME remains a master override."""
        monkeypatch.delenv("MULTI_REPO_MODE", raising=False)
        monkeypatch.setenv("COLLECTION_NAME", "codebase")
        ws = importlib.reload(ws_module)

        assert ws.get_collection_name("my-repo_old") == "codebase_old"

    def test_multi_repo_workspace_level_env_override_still_applies(self, ws_module, monkeypatch):
        """When repo_name is None, env override should still apply even in multi-repo mode."""
        monkeypatch.setenv("MULTI_REPO_MODE", "1")
        monkeypatch.setenv("COLLECTION_NAME", "codebase")
        ws = importlib.reload(ws_module)

        assert ws.get_collection_name(None) == "codebase"


# ============================================================================
# Tests: Environment Variable Helpers
# ============================================================================
class TestEnvHelpers:
    """Tests for environment-based feature flags."""

    def test_is_multi_repo_mode_disabled_by_default(self, ws_module, monkeypatch):
        """Multi-repo mode is disabled by default."""
        monkeypatch.delenv("MULTI_REPO_MODE", raising=False)
        ws = importlib.reload(ws_module)
        assert ws.is_multi_repo_mode() is False

    def test_is_multi_repo_mode_enabled(self, ws_module, monkeypatch):
        """Multi-repo mode can be enabled."""
        for val in ["1", "true", "yes", "on", "TRUE", "ON"]:
            monkeypatch.setenv("MULTI_REPO_MODE", val)
            ws = importlib.reload(ws_module)
            assert ws.is_multi_repo_mode() is True, f"Failed for {val}"

    def test_logical_repo_reuse_disabled_by_default(self, ws_module, monkeypatch):
        """Logical repo reuse is disabled by default."""
        monkeypatch.delenv("LOGICAL_REPO_REUSE", raising=False)
        ws = importlib.reload(ws_module)
        assert ws.logical_repo_reuse_enabled() is False

    def test_logical_repo_reuse_enabled(self, ws_module, monkeypatch):
        """Logical repo reuse can be enabled."""
        monkeypatch.setenv("LOGICAL_REPO_REUSE", "1")
        ws = importlib.reload(ws_module)
        assert ws.logical_repo_reuse_enabled() is True

    def test_is_staging_enabled_disabled_by_default(self, ws_module, monkeypatch):
        """Staging is disabled by default."""
        monkeypatch.delenv("CTXCE_STAGING_ENABLED", raising=False)
        ws = importlib.reload(ws_module)
        assert ws.is_staging_enabled() is False


# ============================================================================
# Tests: State Path Helpers
# ============================================================================
class TestStatePathHelpers:
    """Tests for state path resolution functions."""

    def test_resolve_workspace_root_from_env(self, ws_module, monkeypatch, tmp_path):
        """Workspace root is resolved from WORKSPACE_PATH."""
        ws_root = tmp_path / "custom-root"
        ws_root.mkdir()
        monkeypatch.setenv("WORKSPACE_PATH", str(ws_root))
        ws = importlib.reload(ws_module)

        assert ws._resolve_workspace_root() == str(ws_root)

    def test_resolve_workspace_root_fallback_watch_root(self, ws_module, monkeypatch, tmp_path):
        """Falls back to WATCH_ROOT when WORKSPACE_PATH not set."""
        ws_root = tmp_path / "watch-root"
        ws_root.mkdir()
        monkeypatch.delenv("WORKSPACE_PATH", raising=False)
        monkeypatch.setenv("WATCH_ROOT", str(ws_root))
        ws = importlib.reload(ws_module)

        assert ws._resolve_workspace_root() == str(ws_root)

    def test_get_state_path(self, ws_module, tmp_path):
        """State path is workspace/.codebase/state.json."""
        ws_path = tmp_path / "my-workspace"
        ws_path.mkdir()

        state_path = ws_module._get_state_path(str(ws_path))

        assert state_path.name == "state.json"
        assert state_path.parent.name == ".codebase"

    def test_ensure_state_dir_creates_codebase_dir(self, ws_module, tmp_path):
        """_ensure_state_dir creates .codebase directory."""
        ws_path = tmp_path / "new-workspace"
        ws_path.mkdir()

        state_path = ws_module._ensure_state_dir(str(ws_path))

        assert state_path.parent.exists()
        assert state_path.parent.name == ".codebase"


# ============================================================================
# Tests: State Read/Write Operations
# ============================================================================
class TestStateReadWrite:
    """Tests for get_workspace_state and update_workspace_state."""

    def test_get_workspace_state_creates_new_state(self, ws_module, tmp_path, monkeypatch):
        """get_workspace_state creates state file if it doesn't exist."""
        ws_path = tmp_path / "fresh-workspace"
        ws_path.mkdir()
        monkeypatch.setenv("WORKSPACE_PATH", str(ws_path))
        ws = importlib.reload(ws_module)

        state = ws.get_workspace_state(str(ws_path))

        assert "qdrant_collection" in state
        assert "created_at" in state
        assert "updated_at" in state
        assert state["indexing_status"]["state"] == "idle"

        # Verify file was created
        state_file = ws_path / ".codebase" / "state.json"
        assert state_file.exists()

    def test_get_workspace_state_reads_existing_state(self, ws_module, tmp_path, monkeypatch):
        """get_workspace_state reads existing state file."""
        ws_path = tmp_path / "existing-workspace"
        ws_path.mkdir()
        (ws_path / ".codebase").mkdir()

        existing_state = {
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "qdrant_collection": "test-collection",
            "indexing_status": {"state": "watching"},
        }
        state_file = ws_path / ".codebase" / "state.json"
        state_file.write_text(json.dumps(existing_state))

        monkeypatch.setenv("WORKSPACE_PATH", str(ws_path))
        ws = importlib.reload(ws_module)

        state = ws.get_workspace_state(str(ws_path))

        assert state["qdrant_collection"] == "test-collection"
        assert state["indexing_status"]["state"] == "watching"

    def test_update_workspace_state_merges_updates(self, ws_module, tmp_path, monkeypatch):
        """update_workspace_state merges updates into existing state."""
        ws_path = tmp_path / "update-workspace"
        ws_path.mkdir()
        monkeypatch.setenv("WORKSPACE_PATH", str(ws_path))
        ws = importlib.reload(ws_module)

        # Create initial state
        ws.get_workspace_state(str(ws_path))

        # Update state
        updated = ws.update_workspace_state(
            str(ws_path),
            {"indexing_status": {"state": "indexing"}}
        )

        assert updated["indexing_status"]["state"] == "indexing"
        assert "qdrant_collection" in updated  # Original field preserved


# ============================================================================
# Tests: File Locking
# ============================================================================
class TestFileLocking:
    """Tests for per-file locking mechanism."""

    def test_is_file_locked_returns_false_when_no_lock(self, ws_module, tmp_path):
        """is_file_locked returns False when no lock exists."""
        fake_file = str(tmp_path / "nonexistent.py")
        assert ws_module.is_file_locked(fake_file) is False

    def test_file_indexing_lock_context_manager(self, ws_module, tmp_path, monkeypatch):
        """file_indexing_lock creates and removes lock file."""
        # Use tmp_path for lock files
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir()
        monkeypatch.setattr(ws_module, "_FILE_LOCKS_DIR", lock_dir)

        test_file = str(tmp_path / "test.py")

        # Lock should not exist before
        assert ws_module.is_file_locked(test_file) is False

        # Lock should exist during context
        with ws_module.file_indexing_lock(test_file):
            lock_path = ws_module._get_file_lock_path(test_file)
            assert lock_path.exists()

        # Lock should be removed after context
        assert not lock_path.exists()

    def test_file_indexing_lock_prevents_double_lock(self, ws_module, tmp_path, monkeypatch):
        """file_indexing_lock raises FileExistsError if already locked."""
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir()
        monkeypatch.setattr(ws_module, "_FILE_LOCKS_DIR", lock_dir)

        test_file = str(tmp_path / "test.py")

        with ws_module.file_indexing_lock(test_file):
            with pytest.raises(FileExistsError):
                with ws_module.file_indexing_lock(test_file):
                    pass  # Should not reach here


# ============================================================================
# Tests: Compute Logical Repo ID
# ============================================================================
class TestLogicalRepoId:
    """Tests for compute_logical_repo_id function."""

    def test_compute_logical_repo_id_fs_fallback(self, ws_module, tmp_path, monkeypatch):
        """compute_logical_repo_id uses fs: prefix when not a git repo."""
        # Suppress git detection
        def _no_git(*args, **kwargs):
            return None

        monkeypatch.setattr(ws_module, "_detect_git_common_dir", _no_git)

        ws_path = tmp_path / "not-a-repo"
        ws_path.mkdir()

        lrid = ws_module.compute_logical_repo_id(str(ws_path))

        assert lrid.startswith("fs:")
        assert len(lrid) == 3 + 16  # "fs:" + 16 char hash

    def test_compute_logical_repo_id_consistent(self, ws_module, tmp_path, monkeypatch):
        """compute_logical_repo_id returns consistent results."""
        def _no_git(*args, **kwargs):
            return None

        monkeypatch.setattr(ws_module, "_detect_git_common_dir", _no_git)

        ws_path = tmp_path / "consistent-repo"
        ws_path.mkdir()

        lrid1 = ws_module.compute_logical_repo_id(str(ws_path))
        lrid2 = ws_module.compute_logical_repo_id(str(ws_path))

        assert lrid1 == lrid2


# ============================================================================
# Tests: Atomic Write
# ============================================================================
class TestAtomicWrite:
    """Tests for _atomic_write_state function."""

    def test_atomic_write_creates_valid_json(self, ws_module, tmp_path):
        """_atomic_write_state creates valid JSON file."""
        state_file = tmp_path / "state.json"
        state = {
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "qdrant_collection": "test",
        }

        ws_module._atomic_write_state(state_file, state)

        assert state_file.exists()
        loaded = json.loads(state_file.read_text())
        assert loaded["qdrant_collection"] == "test"

    def test_atomic_write_replaces_existing_file(self, ws_module, tmp_path):
        """_atomic_write_state atomically replaces existing file."""
        state_file = tmp_path / "state.json"

        # Write initial state
        state_file.write_text('{"old": "data"}')

        # Overwrite with new state
        new_state = {"new": "state"}
        ws_module._atomic_write_state(state_file, new_state)

        loaded = json.loads(state_file.read_text())
        assert "new" in loaded
        assert "old" not in loaded


# ============================================================================
# Tests: Constants
# ============================================================================
class TestConstants:
    """Tests for module constants."""

    def test_state_dirname(self, ws_module):
        """STATE_DIRNAME is .codebase."""
        assert ws_module.STATE_DIRNAME == ".codebase"

    def test_state_filename(self, ws_module):
        """STATE_FILENAME is state.json."""
        assert ws_module.STATE_FILENAME == "state.json"

    def test_placeholder_collection_names(self, ws_module):
        """PLACEHOLDER_COLLECTION_NAMES contains expected values."""
        assert "" in ws_module.PLACEHOLDER_COLLECTION_NAMES
        assert "default-collection" in ws_module.PLACEHOLDER_COLLECTION_NAMES
        assert "my-collection" in ws_module.PLACEHOLDER_COLLECTION_NAMES
