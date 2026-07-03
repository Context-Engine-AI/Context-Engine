#!/usr/bin/env python3
"""
Memory Backup Utility for Qdrant Collections

Exports memories (non-code points) from Qdrant collections to JSON for backup purposes.
Memories are identified as points without file path metadata - typically user-added notes,
context, or other information that's not tied to specific code files.

Usage:
    python scripts/memory_backup.py --collection test-repo-58ecbbc8 --output memories_backup.json
    python scripts/memory_backup.py --collection test-repo-58ecbbc8 --output memories_backup_$(date +%Y%m%d).json
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
    from qdrant_client.models import Filter, FieldCondition, MatchValue
except ImportError:
    print("ERROR: qdrant-client not installed. Install with: pip install qdrant-client")
    sys.exit(1)


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client with environment configuration."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")

    return QdrantClient(url=qdrant_url, api_key=api_key or None)


def is_memory_point(payload: Dict[str, Any]) -> bool:
    """
    Determine if a point is a memory (user-added) rather than code-indexed content.

    Memory points typically:
    - Have no 'path' in metadata (not tied to a file)
    - May have 'source' set to 'memory'
    - Have 'content' field that's not extracted from code

    Args:
        payload: Point payload from Qdrant

    Returns:
        True if this appears to be a memory point, False if it's code content
    """
    if not payload:
        return False

    metadata = payload.get("metadata", {})

    # Primary indicator: no file path means it's likely a memory
    if not metadata.get("path"):
        return True

    # Secondary indicator: explicit source marking
    if metadata.get("source") == "memory":
        return True

    # Tertiary: content-based heuristics
    content = payload.get("information", "")
    if content and not metadata.get("language") and not metadata.get("kind"):
        # Content without language/kind metadata is likely user-added
        return True

    return False


def export_memories(
    collection_name: str,
    output_file: str,
    client: Optional[QdrantClient] = None,
    include_vectors: bool = True,
    batch_size: int = 1000
) -> Dict[str, Any]:
    """
    Export memories from a Qdrant collection to JSON.

    Args:
        collection_name: Qdrant collection name
        output_file: Output JSON file path
        client: Qdrant client instance (will create if None)
        include_vectors: Whether to include vector embeddings in backup
        batch_size: Number of points to fetch per request

    Returns:
        Dict with backup statistics
    """
    if client is None:
        client = get_qdrant_client()

    # Verify collection exists
    try:
        collections = client.get_collections().collections
        if collection_name not in [c.name for c in collections]:
            raise ValueError(f"Collection '{collection_name}' not found")
    except Exception as e:
        raise RuntimeError(f"Failed to access Qdrant: {e}")

    print(f"Exporting memories from collection: {collection_name}")
    print(f"Output file: {output_file}")

    # Get all points from collection
    all_points = []
    total_count = 0
    memory_count = 0

    # Use scroll to get all points efficiently
    next_page_offset = None
    while True:
        points, next_page_offset = client.scroll(
            collection_name=collection_name,
            offset=next_page_offset,
            limit=batch_size,
            with_payload=True,
            with_vectors=include_vectors
        )

        if not points:
            break

        all_points.extend(points)
        total_count += len(points)

        # Filter for memory points
        memory_points = []
        for point in points:
            if is_memory_point(point.payload or {}):
                memory_points.append(point)
                memory_count += 1

        print(f"Fetched {len(points)} points (total: {total_count}), found {len(memory_points)} memories (total: {memory_count})")

        if next_page_offset is None:
            break

    if memory_count == 0:
        print("No memories found in collection!")
        return {
            "collection": collection_name,
            "total_points": total_count,
            "memory_count": 0,
            "backup_file": output_file,
            "success": True
        }

    # Prepare backup data
    backup_data = {
        "backup_info": {
            "collection_name": collection_name,
            "export_date": datetime.now().isoformat(),
            "total_points_exported": total_count,
            "memory_points_found": memory_count,
            "include_vectors": include_vectors,
            "vector_dimension": None  # Will be set if vectors included
        },
        "memories": []
    }

    # Process memory points
    for point in all_points:
        if not is_memory_point(point.payload or {}):
            continue

        payload = point.payload or {}
        memory_entry = {
            "id": str(point.id),
            "content": payload.get("information", ""),
            "metadata": payload.get("metadata", {}),
        }

        # Include vector if requested
        if include_vectors and point.vector:
            if hasattr(point.vector, 'tolist'):
                memory_entry["vector"] = point.vector.tolist()
            else:
                memory_entry["vector"] = point.vector

            # Set vector dimension from first memory
            if backup_data["backup_info"]["vector_dimension"] is None:
                vector_data = memory_entry["vector"]
                if isinstance(vector_data, dict):
                    # Named vector format: {"memory": [values]}
                    first_vector = next(iter(vector_data.values()))
                    backup_data["backup_info"]["vector_dimension"] = len(first_vector)
                else:
                    # Direct vector list format
                    backup_data["backup_info"]["vector_dimension"] = len(vector_data)

        backup_data["memories"].append(memory_entry)

    # Write backup file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(backup_data, f, indent=2)

    print(f"✅ Backup completed successfully!")
    print(f"   Total points processed: {total_count}")
    print(f"   Memory points exported: {memory_count}")
    print(f"   Backup file: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    return {
        "collection": collection_name,
        "total_points": total_count,
        "memory_count": memory_count,
        "backup_file": str(output_path),
        "file_size": output_path.stat().st_size,
        "success": True
    }


def list_collections() -> None:
    """List all available Qdrant collections."""
    client = get_qdrant_client()

    try:
        collections = client.get_collections().collections
        print("Available collections:")
        for collection in collections:
            info = client.get_collection(collection.name)
            point_count = info.points_count
            print(f"  - {collection.name} ({point_count:,} points)")
    except Exception as e:
        print(f"Error listing collections: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Backup memories (non-code points) from Qdrant collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --collection test-repo-58ecbbc8 --output memories_backup.json
  %(prog)s --list-collections
  %(prog)s --collection test-repo-58ecbbc8 --output backup_$(date +%Y%m%d_%H%M%S).json --no-vectors
        """
    )

    parser.add_argument(
        "--collection", "-c",
        required=False,
        help="Qdrant collection name to backup memories from"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path for backup"
    )

    parser.add_argument(
        "--list-collections", "-l",
        action="store_true",
        help="List all available collections"
    )

    parser.add_argument(
        "--no-vectors",
        action="store_true",
        help="Don't include vector embeddings in backup (smaller file, requires re-embedding)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of points to fetch per request (default: 1000)"
    )

    args = parser.parse_args()

    if args.list_collections:
        list_collections()
        return

    if not args.collection:
        parser.error("--collection required unless using --list-collections")

    if not args.output:
        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{args.collection}_memories_{timestamp}.json"

    try:
        result = export_memories(
            collection_name=args.collection,
            output_file=args.output,
            include_vectors=not args.no_vectors,
            batch_size=args.batch_size
        )

        if result["success"]:
            print(f"\n🎉 Memory backup completed successfully!")
            if result["memory_count"] == 0:
                print("   (No memories found to backup)")
        else:
            print(f"\n❌ Memory backup failed!")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during backup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()