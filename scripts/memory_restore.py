#!/usr/bin/env python3
"""
Memory Restore Utility for Qdrant Collections

Imports previously backed up memories into Qdrant collections.
Can restore to existing collections (append) or new ones.
Supports re-embedding memories if vectors were not included in backup.

Usage:
    python scripts/memory_restore.py --backup memories_backup.json --collection test-repo-58ecbbc8
    python scripts/memory_restore.py --backup memories_backup.json --collection new-test-repo --embedding-model BAAI/bge-large-en-v1.5
    python scripts/memory_restore.py --backup memories_backup.json --collection new-collection --new-collection
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, HnswConfigDiff
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}")
    print("Install with: pip install qdrant-client fastembed")
    sys.exit(1)

# Use embedder factory for Qwen3 support; fallback to direct fastembed
try:
    from scripts.embedder import get_embedding_model as _get_embedding_model
    _EMBEDDER_FACTORY = True
except ImportError:
    _EMBEDDER_FACTORY = False
    try:
        from fastembed import TextEmbedding
    except ImportError:
        TextEmbedding = None  # type: ignore


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client with environment configuration."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")

    return QdrantClient(url=qdrant_url, api_key=api_key or None)


def get_embedding_model(model_name: str):
    """Initialize embedding model with Qwen3 support via embedder factory."""
    # Try centralized embedder factory first (supports Qwen3 feature flag)
    if _EMBEDDER_FACTORY:
        return _get_embedding_model(model_name)
    # Fallback to direct fastembed
    if TextEmbedding is not None:
        try:
            return TextEmbedding(model_name=model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}")
    raise RuntimeError(
        "No embedding model available. Install fastembed: pip install fastembed"
    )


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    vector_dimension: int,
    vector_name: str = "memory"
) -> None:
    """
    Ensure the target collection exists with appropriate vector configuration.

    Args:
        client: Qdrant client instance
        collection_name: Collection name
        vector_dimension: Vector dimensions for memories
        vector_name: Name for the memory vector
    """
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        if collection_name in [c.name for c in collections]:
            print(f"Collection '{collection_name}' already exists")
            return
    except Exception as e:
        print(f"Warning: Could not check collection existence: {e}")

    # Create collection with memory vector
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                vector_name: VectorParams(
                    size=vector_dimension,
                    distance=Distance.COSINE
                )
            },
            hnsw_config=HnswConfigDiff(m=16, ef_construct=256),
        )
        print(f"Created collection '{collection_name}' with {vector_dimension}-dim vectors")
    except Exception as e:
        raise RuntimeError(f"Failed to create collection '{collection_name}': {e}")


def restore_memories(
    backup_file: str,
    collection_name: str,
    client: Optional[QdrantClient] = None,
    embedding_model_name: Optional[str] = None,
    vector_name: str = "memory",
    batch_size: int = 100,
    skip_existing: bool = True,
    skip_collection_creation: bool = False
) -> Dict[str, Any]:
    """
    Restore memories from backup file to Qdrant collection.

    Args:
        backup_file: Path to backup JSON file
        collection_name: Target collection name
        client: Qdrant client instance (will create if None)
        embedding_model_name: Model name for re-embedding (if vectors not in backup)
        vector_name: Name for the memory vector in collection
        batch_size: Number of memories to upload per batch
        skip_existing: Skip memories that already exist in collection
        skip_collection_creation: Skip collection creation (useful when collection is already configured)

    Returns:
        Dict with restore statistics
    """
    if client is None:
        client = get_qdrant_client()

    # Load backup file
    backup_path = Path(backup_file)
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file}")

    try:
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Invalid backup file format: {e}")

    # Validate backup structure
    if "memories" not in backup_data:
        raise ValueError("Invalid backup file: missing 'memories' section")

    memories = backup_data["memories"]
    backup_info = backup_data.get("backup_info", {})

    print(f"Restoring memories from: {backup_file}")
    print(f"Target collection: {collection_name}")
    print(f"Memories in backup: {len(memories)}")

    if backup_info:
        print(f"Original collection: {backup_info.get('collection_name', 'unknown')}")
        print(f"Backup date: {backup_info.get('export_date', 'unknown')}")
        print(f"Vector dimension: {backup_info.get('vector_dimension', 'unknown')}")

    # Determine vector configuration
    vectors_included = backup_info.get("include_vectors", True) and memories and "vector" in memories[0]

    if not vectors_included:
        if not embedding_model_name:
            # Use default model
            embedding_model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

        print(f"Vectors not included in backup, will re-embed with: {embedding_model_name}")
        embedding_model = get_embedding_model(embedding_model_name)

        # Get vector dimension from model
        test_vector = next(embedding_model.embed(["test"])).tolist()
        vector_dimension = len(test_vector)
        print(f"Embedding model vector dimension: {vector_dimension}")
    else:
        # Use dimension from backup
        vector_dimension = backup_info.get("vector_dimension", len(memories[0]["vector"]))
        embedding_model = None
        print(f"Using vectors from backup, dimension: {vector_dimension}")

    # Ensure collection exists (unless skipped)
    if not skip_collection_creation:
        ensure_collection_exists(client, collection_name, vector_dimension, vector_name)
    else:
        print(f"Skipping collection creation for '{collection_name}' (as requested)")

        # Verify collection actually exists when skipping creation
        try:
            client.get_collection(collection_name)
            print(f"Confirmed collection '{collection_name}' exists")
        except Exception:
            raise RuntimeError(f"Collection '{collection_name}' does not exist but creation was skipped")

    # Check for existing memories if skip_existing is True
    existing_ids = set()
    if skip_existing:
        try:
            # Get all existing point IDs
            all_points, _ = client.scroll(
                collection_name=collection_name,
                limit=None,
                with_payload=False,
                with_vectors=False
            )
            existing_ids = {str(point.id) for point in all_points}
            print(f"Found {len(existing_ids)} existing points in collection")
        except Exception as e:
            print(f"Warning: Could not check existing points: {e}")
            skip_existing = False

    # Process and upload memories in batches
    restored_count = 0
    skipped_count = 0
    error_count = 0

    for i in range(0, len(memories), batch_size):
        batch = memories[i:i + batch_size]
        batch_points = []

        for memory in batch:
            raw_id = memory.get("id", "")

            # Qdrant HTTP API expects point IDs to be either an unsigned integer
            # or a UUID string. Backups store IDs as strings, so we convert
            # purely numeric IDs back to integers to match the original type.
            memory_id = raw_id
            try:
                if isinstance(raw_id, str) and raw_id.isdigit():
                    memory_id = int(raw_id)
            except Exception:
                memory_id = raw_id

            # Skip if already exists
            if skip_existing and memory_id in existing_ids:
                skipped_count += 1
                continue

            try:
                # Prepare vector
                if vectors_included:
                    vector = memory.get("vector")
                    if not vector:
                        raise ValueError("Memory missing vector data")
                    # Vector from backup is already in the correct format: {"memory": [values]}
                else:
                    # Re-embed content
                    content = memory.get("content", "")
                    if not content:
                        raise ValueError("Memory missing content for embedding")

                    vector = next(embedding_model.embed([content])).tolist()
                    # For re-embedded vectors, we need to structure them with the vector name
                    vector = {vector_name: vector}

                # Prepare point data
                point_data = {
                    "id": memory_id,
                    "vector": vector,
                    "payload": {
                        "information": memory.get("content", ""),
                        "metadata": memory.get("metadata", {})
                    }
                }

                batch_points.append(point_data)

            except Exception as e:
                print(f"Error processing memory {memory_id}: {e}")
                error_count += 1
                continue

        # Upload batch
        if batch_points:
            try:
                client.upsert(collection_name=collection_name, points=batch_points)
                restored_count += len(batch_points)
                print(f"  Uploaded batch {i//batch_size + 1}: +{len(batch_points)} memories (total: {restored_count})")
            except Exception as e:
                print(f"Error uploading batch {i//batch_size + 1}: {e}")
                error_count += len(batch_points)

    # Final statistics
    print(f"\n Memory restore completed!")
    print(f"   Total memories in backup: {len(memories)}")
    print(f"   Successfully restored: {restored_count}")
    print(f"   Skipped (already exists): {skipped_count}")
    print(f"   Errors: {error_count}")
    print(f"   Target collection: {collection_name}")

    # Verify final count
    try:
        final_count = client.count(collection_name).count
        print(f"   Final collection size: {final_count:,} points")
    except Exception as e:
        print(f"   Warning: Could not get final count: {e}")

    return {
        "collection": collection_name,
        "backup_file": backup_file,
        "total_memories": len(memories),
        "restored": restored_count,
        "skipped": skipped_count,
        "errors": error_count,
        "success": True
    }


def main():
    parser = argparse.ArgumentParser(
        description="Restore memories from backup to Qdrant collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --backup memories_backup.json --collection test-repo-58ecbbc8
  %(prog)s --backup memories_backup.json --collection new-test-repo --embedding-model BAAI/bge-large-en-v1.5
  %(prog)s --backup memories_backup.json --collection new-collection --new-collection --no-skip-existing
        """
    )

    parser.add_argument(
        "--backup", "-b",
        required=True,
        help="Path to backup JSON file"
    )

    parser.add_argument(
        "--collection", "-c",
        required=True,
        help="Target Qdrant collection name"
    )

    parser.add_argument(
        "--embedding-model", "-m",
        help="Embedding model for re-embedding (if vectors not in backup)"
    )

    parser.add_argument(
        "--vector-name",
        default="memory",
        help="Name for the memory vector in collection (default: memory)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of memories to upload per batch (default: 100)"
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Don't skip memories that already exist in collection"
    )

    parser.add_argument(
        "--list-backup-info",
        action="store_true",
        help="Show backup file information without restoring"
    )

    parser.add_argument(
        "--skip-collection-creation",
        action="store_true",
        help="Skip collection creation (useful when collection is already configured by other processes)"
    )

    args = parser.parse_args()

    try:
        # Load backup to show info
        with open(args.backup, 'r') as f:
            backup_data = json.load(f)

        if args.list_backup_info:
            print("Backup Information:")
            print("=" * 50)
            backup_info = backup_data.get("backup_info", {})
            for key, value in backup_info.items():
                print(f"  {key}: {value}")

            memories = backup_data.get("memories", [])
            print(f"  Memory count: {len(memories)}")

            if memories:
                sample = memories[0]
                has_vector = "vector" in sample
                print(f"  Has vectors: {has_vector}")
                if has_vector:
                    vector_dim = len(sample["vector"])
                    print(f"  Vector dimension: {vector_dim}")

            return

        # Restore memories
        result = restore_memories(
            backup_file=args.backup,
            collection_name=args.collection,
            embedding_model_name=args.embedding_model,
            vector_name=args.vector_name,
            batch_size=args.batch_size,
            skip_existing=not args.no_skip_existing,
            skip_collection_creation=args.skip_collection_creation
        )

        if result["success"]:
            print(f"\n  Memory restoration completed successfully!")
        else:
            print(f"\n  Memory restoration failed!")
            sys.exit(1)

    except Exception as e:
        print(f"\n  Error during restoration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()