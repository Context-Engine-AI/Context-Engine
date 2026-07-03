# Indexing Configuration Drift

This document tracks which environment/configuration knobs impact the indexing pipeline, what kind of maintenance they require, and how the admin UI surfaces that drift.

## Snapshot source

`get_indexing_config_snapshot()` (@scripts/workspace_state.py) captures the runtime-configurable indexing knobs. `build_admin_collections_view()` compares the persisted snapshot for each collection against the current runtime snapshot, then classifies the drift using `CONFIG_DRIFT_RULES` (@scripts/indexing_admin.py).

## Drift matrix

| Config key(s) | Effect | Drift class | Notes |
| --- | --- | --- | --- |
| `EMBEDDING_MODEL`, `EMBEDDING_PROVIDER` | Changes vector dimension/model space. | **Recreate** | Collection schema and stored vectors must match the embedding model; `ensure_collection()` recreates the collection when schema differs. |
| `REFRAG_MODE`, `QWEN3_EMBEDDING_ENABLED`, `MINI_VEC_DIM`, `LEX_SPARSE_MODE` | Adds/removes named vectors (mini, sparse) and advanced embedding paths. | **Recreate** | Qdrant cannot add/remove vector names in-place; we back up → delete → recreate the collection to apply these changes. |
| `INDEX_SEMANTIC_CHUNKS`, `INDEX_CHUNK_LINES`, `INDEX_CHUNK_OVERLAP` | Alters semantic chunk sizes/overlaps. | **Reindex** | Requires reprocessing every file so chunks align with the new geometry. |
| `INDEX_MICRO_CHUNKS`, `MICRO_CHUNK_TOKENS`, `MICRO_CHUNK_STRIDE`, `MAX_MICRO_CHUNKS_PER_FILE` | Controls micro-chunk tier and token window. | **Reindex** | Points must be regenerated with the new token windows. |
| `USE_TREE_SITTER`, `INDEX_USE_ENHANCED_AST` | Enables AST-guided segmentation + smart reindex logic. | **Reindex** | Cached symbols and chunk boundaries change; files need reindexing to stay consistent. |

## Action guidance

| Drift class | Required action | UI indicator |
| --- | --- | --- |
| **Recreate** | Use “Recreate” to drop/rebuild the Qdrant collection (or run staging rebuild). | “maintenance needed – recreate” (red) with list of drift keys. |
| **Reindex** | Run “Reindex” or a staging rebuild to reprocess files with the new chunk/AST settings. The reindex action now clears file-hash and symbol caches first so every file reprocesses even if unchanged. | “config drift – reindex” (amber) with drift keys. |

## Future enhancements

* Env-hash-aware cache invalidation so smart reindex won’t reuse stale symbol snapshots when drift occurs.
* Automated “bulk reindex” job for reindex-only drift classes, so the warning clears once the sweep completes.
* Optional suppression of warnings for knobs proven to be safe hot applies once supporting automation exists.
* Reindex can clean up old symbol caches to free disk space and remove stale chunks.
* File-level reindex only runs when a file changes, so untouched files retain the previous chunking; without env-hash-triggered sweeping or cache invalidation, consistency currently requires a full recreate or manual reindex of every file.

