"""Core building blocks for the watch_index entrypoint.

Modules:
    config: shared configuration constants and logger
    routing: repo/collection detection helpers
    queue: debounced change queue implementation
    handler: watchdog event handler logic
    processor: batch processing + ingest handoff
    rename: move/rename fast-path helpers
    pseudo: background pseudo backfill worker
"""

from . import config, routing, queue, handler, processor, rename, pseudo

__all__ = [
    "config",
    "routing",
    "queue",
    "handler",
    "processor",
    "rename",
    "pseudo",
]
