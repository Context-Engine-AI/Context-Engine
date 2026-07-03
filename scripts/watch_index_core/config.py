"""Shared configuration and logging helpers for watch_index."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from scripts.logger import get_logger


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def build_logger():
    """Create a logger, falling back to logging.getLogger when the main logger fails."""
    try:
        return get_logger("scripts.watch_index")
    except Exception:  # pragma: no cover - fallback for logger import issues
        import logging

        return logging.getLogger("scripts.watch_index")


LOGGER = build_logger()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
ROOT = Path(os.environ.get("WATCH_ROOT", "/work")).resolve()

# Debounce interval for file system events
DELAY_SECS = float(os.environ.get("WATCH_DEBOUNCE_SECS", "1.0"))


def default_collection_name() -> str:
    """Base fallback for collection name before runtime resolution."""
    return os.environ.get("COLLECTION_NAME", "codebase")
