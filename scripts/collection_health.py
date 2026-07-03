#!/usr/bin/env python3
"""
Collection health monitoring and self-healing for cache/collection sync issues.

Detects when the local cache is out of sync with the actual Qdrant collection
and triggers corrective actions (cache clear + reindex).
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.workspace_state import (
    _read_cache,
    _write_cache,
    get_workspace_state,
    update_workspace_state,
    clear_symbol_cache,
    is_multi_repo_mode,
)

logger = logging.getLogger(__name__)


def get_cached_files_count(workspace_path: str) -> int:
    """Return the number of files tracked in the local cache."""
    try:
        cache = _read_cache(workspace_path)
        file_hashes = cache.get("file_hashes", {})
        return len(file_hashes)
    except Exception as e:
        logger.warning(f"Failed to read cache: {e}")
        return 0


def get_collection_points_count(collection_name: str, qdrant_url: Optional[str] = None) -> int:
    """Return the number of points in the Qdrant collection."""
    try:
        from qdrant_client import QdrantClient
        
        url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        api_key = os.environ.get("QDRANT_API_KEY")
        
        client = QdrantClient(
            url=url,
            api_key=api_key or None,
            timeout=int(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )
        
        result = client.count(collection_name=collection_name, exact=True)
        return int(getattr(result, "count", 0))
    except Exception as e:
        logger.warning(f"Failed to get collection count: {e}")
        return -1


def get_unique_files_in_collection(collection_name: str, qdrant_url: Optional[str] = None) -> int:
    """Return the number of unique files (distinct paths) in the collection."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client import models
        
        url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        api_key = os.environ.get("QDRANT_API_KEY")
        
        client = QdrantClient(
            url=url,
            api_key=api_key or None,
            timeout=int(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )
        
        # Scroll through all points and collect unique paths
        unique_paths = set()
        offset = None
        batch_size = 100
        
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
            )
            
            if not points:
                break
                
            for point in points:
                try:
                    payload = point.payload or {}
                    metadata = payload.get("metadata", {})
                    path = metadata.get("path")
                    if path:
                        unique_paths.add(str(path))
                except Exception:
                    continue
            
            if offset is None:
                break
        
        return len(unique_paths)
    except Exception as e:
        logger.warning(f"Failed to count unique files: {e}")
        return -1


def clear_cache(workspace_path: str) -> bool:
    """Clear the local file hash cache."""
    try:
        cache = {"file_hashes": {}, "updated_at": ""}
        _write_cache(workspace_path, cache)
        logger.info(f"Cleared cache for workspace: {workspace_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return False


def clear_repo_cache(repo_name: str) -> bool:
    """Clear the local file hash cache for a specific repo (multi-repo mode)."""
    try:
        from scripts.workspace_state import _get_repo_state_dir, CACHE_FILENAME
        from datetime import datetime

        state_dir = _get_repo_state_dir(repo_name)
        cache_path = state_dir / CACHE_FILENAME

        if cache_path.exists():
            cache = {"file_hashes": {}, "updated_at": datetime.now().isoformat()}
            with open(cache_path, "w", encoding="utf-8") as f:
                import json
                json.dump(cache, f, indent=2)
            logger.info(f"Cleared cache for repo: {repo_name}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to clear repo cache for {repo_name}: {e}")
        return False


def clear_indexing_caches(workspace_path: str, repo_name: Optional[str] = None) -> Dict[str, Any]:
    """Clear file-hash and symbol caches for the workspace/repo."""
    result: Dict[str, Any] = {
        "file_hash_cache": False,
        "symbol_cache_dirs": 0,
    }

    multi_repo = False
    try:
        multi_repo = is_multi_repo_mode()
    except Exception:
        multi_repo = False

    try:
        if multi_repo and repo_name:
            result["file_hash_cache"] = clear_repo_cache(repo_name)
        else:
            result["file_hash_cache"] = clear_cache(workspace_path)
    except Exception as e:
        result["file_hash_error"] = str(e)

    if clear_symbol_cache:
        try:
            cleared_dirs = clear_symbol_cache(workspace_path, repo_name)
            result["symbol_cache_dirs"] = cleared_dirs
        except Exception as e:
            result["symbol_cache_error"] = str(e)
    else:
        result["symbol_cache_error"] = "clear_symbol_cache unavailable"

    return result


def get_repo_cached_files_count(repo_name: str) -> int:
    """Return the number of files tracked in a repo's cache (multi-repo mode)."""
    try:
        from scripts.workspace_state import _get_repo_state_dir, CACHE_FILENAME
        import json

        state_dir = _get_repo_state_dir(repo_name)
        cache_path = state_dir / CACHE_FILENAME

        if not cache_path.exists():
            return 0

        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        return len(cache.get("file_hashes", {}))
    except Exception as e:
        logger.warning(f"Failed to read repo cache for {repo_name}: {e}")
        return 0


def auto_heal_multi_repo(
    workspace_path: str,
    qdrant_url: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Health check for multi-repo mode. Checks each repo's collection.

    Only clears cache when collection is EMPTY but cache has entries.
    This is the safest possible check - avoids clearing cache unnecessarily.

    Returns dict with per-repo results.
    """
    from scripts.workspace_state import is_multi_repo_mode, get_collection_name

    results = {"repos_checked": 0, "repos_healed": 0, "details": {}}

    if not is_multi_repo_mode():
        results["skipped"] = "not in multi-repo mode"
        return results

    # Find all repos under workspace
    ws = Path(workspace_path)
    repos = []
    for entry in ws.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            # Check if it looks like a repo (has .git or files)
            if (entry / ".git").exists() or any(entry.iterdir()):
                repos.append(entry.name)

    for repo_name in repos:
        try:
            # Get collection name for this repo
            repo_path = ws / repo_name
            collection = get_collection_name(str(repo_path))
            if not collection:
                continue

            cached_count = get_repo_cached_files_count(repo_name)
            if cached_count == 0:
                # No cache entries, nothing to validate
                continue

            points_count = get_collection_points_count(collection, qdrant_url)
            results["repos_checked"] += 1

            repo_result = {
                "collection": collection,
                "cached_files": cached_count,
                "collection_points": points_count,
                "action": "none",
            }

            # ONLY clear if collection is completely empty
            # This is ultra-conservative - won't clear for partial issues
            if points_count == 0 and cached_count > 0:
                repo_result["issue"] = f"Collection empty but cache has {cached_count} files"
                if not dry_run:
                    if clear_repo_cache(repo_name):
                        repo_result["action"] = "cleared_cache"
                        results["repos_healed"] += 1
                else:
                    repo_result["action"] = "would_clear_cache"

            results["details"][repo_name] = repo_result

        except Exception as e:
            results["details"][repo_name] = {"error": str(e)}

    return results


def detect_collection_health(
    workspace_path: str,
    collection_name: str,
    qdrant_url: Optional[str] = None,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Detect cache/collection sync issues.
    
    Returns a dict with:
    - healthy: bool
    - cached_files: int
    - collection_points: int
    - unique_files_in_collection: int
    - issue: Optional[str] - description of the problem
    - recommendation: Optional[str] - suggested fix
    """
    cached_count = get_cached_files_count(workspace_path)
    points_count = get_collection_points_count(collection_name, qdrant_url)
    unique_files = get_unique_files_in_collection(collection_name, qdrant_url)
    
    result = {
        "healthy": True,
        "cached_files": cached_count,
        "collection_points": points_count,
        "unique_files_in_collection": unique_files,
        "issue": None,
        "recommendation": None,
    }
    
    # Check 1: Collection is empty but cache has entries
    if points_count == 0 and cached_count > 0:
        result["healthy"] = False
        result["issue"] = f"Collection is empty but cache has {cached_count} files"
        result["recommendation"] = "Clear cache and force reindex"
        return result
    
    # Check 2: Unique files in collection is way less than cached files
    if unique_files >= 0 and cached_count > 0:
        ratio = unique_files / cached_count if cached_count > 0 else 0
        if ratio < threshold:
            result["healthy"] = False
            result["issue"] = (
                f"Cache has {cached_count} files but collection only has {unique_files} "
                f"unique files ({ratio:.1%} < {threshold:.0%} threshold)"
            )
            result["recommendation"] = "Clear cache and force reindex"
            return result
    
    # Check 3: Collection has points but no unique files detected (metadata issue)
    if points_count > 0 and unique_files == 0:
        result["healthy"] = False
        result["issue"] = f"Collection has {points_count} points but no valid file paths in metadata"
        result["recommendation"] = "Recreate collection with proper metadata"
        return result
    
    return result


def auto_heal_if_needed(
    workspace_path: str,
    collection_name: str,
    qdrant_url: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Detect and automatically fix cache/collection sync issues.
    
    Returns a dict with:
    - action_taken: str
    - health_check: Dict (from detect_collection_health)
    """
    health = detect_collection_health(workspace_path, collection_name, qdrant_url)
    
    result = {
        "action_taken": "none",
        "health_check": health,
    }
    
    if not health["healthy"]:
        logger.warning(f"Collection health issue detected: {health['issue']}")
        logger.info(f"Recommendation: {health['recommendation']}")
        
        if not dry_run:
            if "Clear cache" in health["recommendation"]:
                if clear_cache(workspace_path):
                    result["action_taken"] = "cleared_cache"
                    logger.info("Cache cleared. Reindex required.")
                else:
                    result["action_taken"] = "clear_cache_failed"
        else:
            result["action_taken"] = "dry_run"
            logger.info("Dry run mode - no action taken")
    else:
        logger.info("Collection health check passed")
    
    return result


def main():
    """CLI for health checking and healing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check and heal collection health")
    parser.add_argument(
        "--workspace",
        default=os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work",
        help="Workspace path (default: WATCH_ROOT or /work)",
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("COLLECTION_NAME", "codebase"),
        help="Collection name (default: COLLECTION_NAME env or codebase)",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL (default: QDRANT_URL env or http://localhost:6333)",
    )
    parser.add_argument(
        "--auto-heal",
        action="store_true",
        help="Automatically fix issues (clear cache if needed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check health but don't take action",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    if args.auto_heal:
        result = auto_heal_if_needed(
            args.workspace,
            args.collection,
            args.qdrant_url,
            dry_run=args.dry_run,
        )
        print(f"\nAction taken: {result['action_taken']}")
    else:
        health = detect_collection_health(
            args.workspace,
            args.collection,
            args.qdrant_url,
        )
        print(f"\nHealth check results:")
        print(f"  Healthy: {health['healthy']}")
        print(f"  Cached files: {health['cached_files']}")
        print(f"  Collection points: {health['collection_points']}")
        print(f"  Unique files in collection: {health['unique_files_in_collection']}")
        if not health['healthy']:
            print(f"  Issue: {health['issue']}")
            print(f"  Recommendation: {health['recommendation']}")


if __name__ == "__main__":
    main()

