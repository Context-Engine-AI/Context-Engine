"""Test error handling for workspace_state cache operations in watch_index_core."""
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_handler_invalidate_cache_handles_errors(monkeypatch, tmp_path):
    """Verify IndexHandler._invalidate_cache gracefully handles workspace_state errors."""
    handler_mod = importlib.import_module("scripts.watch_index_core.handler")
    ws_mod = importlib.import_module("scripts.workspace_state")
    
    # Mock workspace_state functions to raise errors
    monkeypatch.setattr(ws_mod, "remove_cached_file", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(ws_mod, "remove_cached_symbols", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    
    # Create a minimal handler
    queue = MagicMock()
    client = MagicMock()
    h = handler_mod.IndexHandler(tmp_path, queue, client, "test-coll")
    
    # Should not raise despite errors
    result = h._invalidate_cache(tmp_path / "test.py")
    assert result is None  # No repo_name when not in multi-repo mode


def test_processor_handles_cache_read_errors(monkeypatch, tmp_path):
    """Verify processor gracefully handles get_cached_file_hash errors."""
    proc_mod = importlib.import_module("scripts.watch_index_core.processor")
    ws_mod = importlib.import_module("scripts.workspace_state")
    
    # Mock to raise error
    monkeypatch.setattr(ws_mod, "get_cached_file_hash", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content", encoding="utf-8")
    
    # Exercise _read_text_and_sha1 - should handle errors gracefully and return content + hash
    text, sha1 = proc_mod._read_text_and_sha1(test_file)
    assert text == "test content"
    assert sha1 is not None and len(sha1) == 40  # SHA1 hex length


def test_handler_move_event_handles_cache_errors(monkeypatch, tmp_path):
    """Verify on_moved handles cache errors gracefully by completing without raising."""
    handler_mod = importlib.import_module("scripts.watch_index_core.handler")
    ws_mod = importlib.import_module("scripts.workspace_state")
    
    # Mock workspace_state to raise
    monkeypatch.setattr(ws_mod, "get_cached_file_hash", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(ws_mod, "set_cached_file_hash", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(ws_mod, "remove_cached_file", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(ws_mod, "remove_cached_symbols", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    
    # Create source and dest files
    src_file = tmp_path / "src.py"
    dest_file = tmp_path / "dest.py"
    src_file.write_text("# code", encoding="utf-8")
    dest_file.write_text("# code", encoding="utf-8")
    
    # Create handler and mock event
    queue = MagicMock()
    client = MagicMock()
    client.scroll = MagicMock(return_value=([], None))  # No points to rename
    h = handler_mod.IndexHandler(tmp_path, queue, client, "test-coll")
    
    # Create mock move event
    event = MagicMock()
    event.is_directory = False
    event.src_path = str(src_file)
    event.dest_path = str(dest_file)
    
    # Should complete without raising despite cache errors
    result = h.on_moved(event)
    assert result is None  # on_moved returns None


def test_processor_handles_cache_remove_errors(monkeypatch, tmp_path):
    """Verify processor handles remove_cached_file errors when processing deletes."""
    proc_mod = importlib.import_module("scripts.watch_index_core.processor")
    ws_mod = importlib.import_module("scripts.workspace_state")
    
    # Mock to raise
    monkeypatch.setattr(ws_mod, "remove_cached_file", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    
    missing = tmp_path / "missing.py"
    assert not missing.exists()

    # Mock other dependencies
    monkeypatch.setattr(proc_mod, "_detect_repo_for_file", lambda p: tmp_path)
    monkeypatch.setattr(proc_mod, "_get_collection_for_file", lambda p: "coll")
    monkeypatch.setattr(proc_mod, "_set_status_indexing", lambda *a, **k: None)
    monkeypatch.setattr(proc_mod, "persist_indexing_config", lambda *a, **k: None)
    monkeypatch.setattr(proc_mod, "update_indexing_status", lambda *a, **k: None)
    monkeypatch.setattr(proc_mod, "get_workspace_state", lambda *a, **k: {})
    monkeypatch.setattr(proc_mod, "is_staging_enabled", lambda: False)
    monkeypatch.setattr(proc_mod, "_log_activity", lambda *a, **k: None)
    monkeypatch.setattr(proc_mod, "_extract_repo_name_from_path", lambda *_: "repo")

    # _process_paths should complete without raising despite cache errors
    # This exercises the delete path when file doesn't exist
    proc_mod._process_paths(
        [missing],
        client=None,
        model=None,
        vector_name="vec",
        model_dim=1,
        workspace_path=str(tmp_path),
    )
    # If we get here without exception, the error was handled gracefully



def test_watch_index_core_config_root_dir_matches_project_root():
    cfg_mod = importlib.import_module("scripts.watch_index_core.config")
    expected = Path(cfg_mod.__file__).resolve().parents[2]
    assert Path(cfg_mod.ROOT_DIR).resolve() == expected
    assert str(expected) in sys.path


def test_processor_delete_clears_cache_even_without_client(monkeypatch, tmp_path):
    proc_mod = importlib.import_module("scripts.watch_index_core.processor")

    missing = tmp_path / "missing.py"
    assert not missing.exists()

    monkeypatch.setattr(proc_mod, "_detect_repo_for_file", lambda p: tmp_path)
    monkeypatch.setattr(proc_mod, "_get_collection_for_file", lambda p: "coll")
    monkeypatch.setattr(proc_mod, "_set_status_indexing", lambda *a, **k: None)
    monkeypatch.setattr(proc_mod, "persist_indexing_config", lambda *a, **k: None)
    monkeypatch.setattr(proc_mod, "update_indexing_status", lambda *a, **k: None)
    monkeypatch.setattr(proc_mod, "get_workspace_state", lambda *a, **k: {})
    monkeypatch.setattr(proc_mod, "is_staging_enabled", lambda: False)
    monkeypatch.setattr(proc_mod, "_log_activity", lambda *a, **k: None)
    monkeypatch.setattr(proc_mod, "_extract_repo_name_from_path", lambda *_: "repo")

    remove_mock = MagicMock()
    monkeypatch.setattr(proc_mod, "remove_cached_file", remove_mock)

    proc_mod._process_paths(
        [missing],
        client=None,
        model=None,
        vector_name="vec",
        model_dim=1,
        workspace_path=str(tmp_path),
    )

    remove_mock.assert_called_once_with(str(missing), "repo")
