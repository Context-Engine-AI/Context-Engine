#!/usr/bin/env python3
"""
Rerank Training Event Logger.

Logs training events (query, candidates, scores, collection) to a file for
background processing. This keeps the MCP hot path fast and deterministic.

Features:
- Time-sharded files (hourly) to avoid giant files and enable parallel processing
- Configurable sampling rate to reduce volume at high QPS
- Events written as newline-delimited JSON (NDJSON) for streaming reads
"""

import json
import os
import random
import time
import threading
try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Event log configuration
RERANK_EVENTS_DIR = os.environ.get("RERANK_EVENTS_DIR", "/tmp/rerank_events")
RERANK_EVENTS_ENABLED = str(os.environ.get("RERANK_EVENTS_ENABLED", "1")).strip().lower() in {
    "1", "true", "yes", "on"
}
# Sampling rate: 0.5 = log 50% of events (balance volume vs learning signal)
RERANK_EVENTS_SAMPLE_RATE = float(os.environ.get("RERANK_EVENTS_SAMPLE_RATE", "0.5"))
# Retention: files older than this many days can be cleaned up (0 = keep forever)
RERANK_EVENTS_RETENTION_DAYS = int(os.environ.get("RERANK_EVENTS_RETENTION_DAYS", "0"))

# Thread-safe write lock (per-file locks for better concurrency)
_WRITE_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()


def _get_write_lock(file_key: str) -> threading.Lock:
    """Get or create a write lock for a specific file."""
    with _LOCKS_LOCK:
        if file_key not in _WRITE_LOCKS:
            _WRITE_LOCKS[file_key] = threading.Lock()
        return _WRITE_LOCKS[file_key]


def _ensure_events_dir() -> Path:
    """Ensure events directory exists."""
    events_dir = Path(RERANK_EVENTS_DIR)
    events_dir.mkdir(parents=True, exist_ok=True)
    return events_dir


def _get_hour_suffix() -> str:
    """Get current hour suffix for time-sharding (YYYYMMDDHH)."""
    return datetime.now(tz=None).strftime("%Y%m%d%H")


def _get_events_file(collection: str, hour_suffix: Optional[str] = None) -> Path:
    """Get events file path for a collection (time-sharded)."""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)
    if hour_suffix is None:
        hour_suffix = _get_hour_suffix()
    return _ensure_events_dir() / f"events_{safe_name}_{hour_suffix}.ndjson"


def log_training_event(
    query: str,
    candidates: List[Dict[str, Any]],
    initial_scores: List[float],
    teacher_scores: Optional[List[float]],
    collection: str,
    metadata: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> bool:
    """
    Log a training event for background processing.

    Args:
        query: The search query
        candidates: List of candidate documents (will extract path, symbol, snippet)
        initial_scores: Initial hybrid search scores
        teacher_scores: ONNX teacher scores (if available)
        collection: Collection name for isolation
        metadata: Optional additional metadata
        force: If True, bypass sampling (always log)

    Returns:
        True if event was logged successfully, False if skipped/disabled
    """
    if not RERANK_EVENTS_ENABLED:
        return False

    # Sampling: only log SAMPLE_RATE fraction of events
    if not force and random.random() > RERANK_EVENTS_SAMPLE_RATE:
        return False

    try:
        # Helper to convert numpy types to native Python for JSON
        def _to_native(v):
            if hasattr(v, "item"):  # numpy scalar
                return v.item()
            return v

        # Extract minimal candidate info (don't store full code)
        candidate_info = []
        for i, c in enumerate(candidates):
            score = initial_scores[i] if i < len(initial_scores) else 0
            info = {
                "path": c.get("path", ""),
                "symbol": c.get("symbol", ""),
                "start_line": c.get("start_line", 0),
                "end_line": c.get("end_line", 0),
                "initial_score": _to_native(score),
            }
            # Include small snippet for learning (truncated)
            snippet = c.get("code") or c.get("snippet") or ""
            if snippet:
                info["snippet"] = snippet[:500]
            candidate_info.append(info)

        event = {
            "ts": time.time(),
            "query": query,
            "collection": collection,
            "candidates": candidate_info,
            "teacher_scores": [_to_native(s) for s in teacher_scores] if teacher_scores is not None and len(teacher_scores) > 0 else None,
            "metadata": metadata or {},
        }

        events_file = _get_events_file(collection)
        file_key = str(events_file)

        # Per-file lock for better concurrency across collections/hours
        lock = _get_write_lock(file_key)
        with lock:
            # Atomic append with file locking (for cross-process safety)
            with open(events_file, "a") as f:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        f.write(json.dumps(event) + "\n")
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                else:
                    # Best-effort fallback (thread lock still prevents intra-process interleaving)
                    f.write(json.dumps(event) + "\n")

        return True

    except Exception:
        return False


def list_event_files(collection: str) -> List[Path]:
    """List all event files for a collection (sorted by time, oldest first)."""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)
    pattern = f"events_{safe_name}_*.ndjson"
    events_dir = _ensure_events_dir()
    files = sorted(events_dir.glob(pattern))
    return files


def read_events(
    collection: str,
    since_ts: float = 0,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Read training events for a collection (across all time-sharded files).

    Args:
        collection: Collection name
        since_ts: Only return events after this timestamp
        limit: Maximum events to return

    Returns:
        List of training events (oldest first)
    """
    event_files = list_event_files(collection)
    if not event_files:
        return []

    events = []
    for events_file in event_files:
        try:
            with open(events_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("ts", 0) > since_ts:
                            events.append(event)
                            if len(events) >= limit:
                                return events
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    return events


def cleanup_old_events(collection: str, max_age_days: int) -> int:
    """
    Remove event files older than max_age_days.

    Args:
        collection: Collection name
        max_age_days: Delete files older than this

    Returns:
        Number of files deleted
    """
    if max_age_days <= 0:
        return 0

    cutoff = time.time() - (max_age_days * 86400)
    deleted = 0

    for events_file in list_event_files(collection):
        try:
            if events_file.stat().st_mtime < cutoff:
                events_file.unlink()
                deleted += 1
        except Exception:
            continue

    return deleted

