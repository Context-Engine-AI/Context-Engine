#!/usr/bin/env python3
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
from datetime import datetime
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import critical functions first
try:
    from scripts.workspace_state import get_collection_name, is_multi_repo_mode
except Exception:
    get_collection_name = None  # type: ignore
    is_multi_repo_mode = None  # type: ignore

# Import other optional functions
try:
    from scripts.workspace_state import log_activity
except Exception:
    log_activity = None  # type: ignore

COLLECTION = os.environ.get("COLLECTION_NAME", "codebase")
# Discover workspace path for state updates (allows subdir indexing)
WS_PATH = os.environ.get("INDEX_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work"

# Multi-repo mode note:
# - When MULTI_REPO_MODE=1 and we are operating at the multi-repo workspace root (/work),
#   there typically is no single "root" collection that should be created/indexed here.
# - Instead, per-repo collections are created/ensured by the indexer/watcher as each repo is
#   discovered, and payload indexes should be created per-collection at that time.
# - This script is still used in single-repo flows (or when targeting a specific workspace path).
#
# IMPORTANT: The broader bootstrap/init job may still run other startup checks (health/ACL/registry
# sync, etc.) via other scripts; this early-exit only affects the single-collection index creation.
if is_multi_repo_mode and is_multi_repo_mode() and WS_PATH == "/work":
    print("Multi-repo mode enabled - skipping root collection creation for /work")
    exit(0)

# Prefer workspace-derived collection names when env value is a placeholder
if 'get_collection_name' in globals() and get_collection_name:
    try:
        resolved = get_collection_name(None)
        if resolved:
            placeholders = {"", "codebase"}
            if COLLECTION in placeholders:
                COLLECTION = resolved
    except Exception:
        pass


try:
    qdrant_timeout = float(os.environ.get("QDRANT_TIMEOUT", "20") or 20)
except Exception:
    qdrant_timeout = 20.0

cli = QdrantClient(url=QDRANT_URL, timeout=qdrant_timeout)

# Check if collection exists first
try:
    cli.get_collection(COLLECTION)
except Exception as e:
    if "doesn't exist" in str(e) or "Not found" in str(e):
        print(f"Collection '{COLLECTION}' doesn't exist yet - skipping index creation")
        print("Indexes will be created when the collection is indexed")
        exit(0)
    raise

# Create keyword indexes for metadata fields (idempotent - safe to call if already exists)
try:
    cli.create_payload_index(
        collection_name=COLLECTION,
        field_name="metadata.language",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
except Exception as e:
    if "already exists" not in str(e).lower():
        print(f"Warning: Could not create language index: {e}")

try:
    cli.create_payload_index(
        collection_name=COLLECTION,
        field_name="metadata.path_prefix",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
except Exception as e:
    if "already exists" not in str(e).lower():
        print(f"Warning: Could not create path_prefix index: {e}")

# Log activity using cleaned workspace_state function
try:
    if log_activity:
        log_activity(
            repo_name=None,
            action="initialized",
            file_path="",
            details={"created_indexes": ["metadata.language", "metadata.path_prefix"]},
        )
except Exception:
    pass

info = cli.get_collection(COLLECTION)
print(info.payload_schema)
