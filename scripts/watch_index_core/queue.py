"""Debounced change queue used by the watcher."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Iterable, List, Set

from .config import DELAY_SECS, LOGGER


class ChangeQueue:
    """Collects file paths and flushes them after a debounce interval."""

    def __init__(self, process_cb: Callable[[List[Path]], None]):
        self._lock = threading.Lock()
        self._paths: Set[Path] = set()
        self._pending: Set[Path] = set()
        self._timer: threading.Timer | None = None
        self._process_cb = process_cb
        # Serialize processing to avoid concurrent use of TextEmbedding/QdrantClient
        self._processing_lock = threading.Lock()

    def add(self, p: Path) -> None:
        with self._lock:
            self._paths.add(p)
            if self._timer is not None:
                try:
                    self._timer.cancel()
                except Exception as exc:
                    LOGGER.error(
                        "Failed to cancel timer in ChangeQueue.add",
                        extra={"error": str(exc)},
                    )
            self._timer = threading.Timer(DELAY_SECS, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        # Grab current batch
        with self._lock:
            paths = list(self._paths)
            self._paths.clear()
            self._timer = None

        # Try to run the processor exclusively; if busy, queue and return
        if not self._processing_lock.acquire(blocking=False):
            with self._lock:
                self._pending.update(paths)
                if self._timer is None:
                    # schedule a follow-up flush to pick up pending when free
                    self._timer = threading.Timer(DELAY_SECS, self._flush)
                    self._timer.daemon = True
                    self._timer.start()
            return
        try:
            # Per-file locking in index_single_file handles indexer/watcher coordination
            todo: Iterable[Path] = paths
            while True:
                try:
                    self._process_cb(list(todo))
                except Exception as exc:
                    # Log processing error via structured logging
                    try:
                        LOGGER.error(
                            "Processing batch failed in ChangeQueue._flush",
                            extra={"error": str(exc), "batch_size": len(list(todo))},
                            exc_info=True,
                        )
                    except Exception as inner_exc:  # pragma: no cover - logging fallback
                        # If logging fails, ensure we don't lose both errors
                        import sys
                        try:
                            print(
                                f"[watcher_error] logging failed: {inner_exc}, "
                                f"original error: {exc}",
                                file=sys.stderr,
                            )
                        except Exception:
                            pass  # Last resort: can't even print
                # drain any pending accumulated during processing
                with self._lock:
                    if not self._pending:
                        break
                    todo = list(self._pending)
                    self._pending.clear()
        finally:
            self._processing_lock.release()


__all__ = ["ChangeQueue"]
