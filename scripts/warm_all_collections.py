#!/usr/bin/env python3
"""
Script to warm all collections in Qdrant
"""
import os
import sys
import subprocess
from qdrant_client import QdrantClient

def main():
    # Get configuration from environment
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    ef = os.environ.get("EF", "256")
    limit = os.environ.get("LIMIT", "3")

    print(f"Connecting to Qdrant at {qdrant_url}")

    # Connect to Qdrant
    client = QdrantClient(url=qdrant_url)

    # Get all collections
    try:
        collections_response = client.get_collections()
        collections = [c.name for c in collections_response.collections]
        print(f"Found collections: {collections}")
    except Exception as e:
        print(f"Error getting collections: {e}")
        sys.exit(1)

    # Warm each collection
    for collection_name in collections:
        print(f"Warming collection: {collection_name}")
        try:
            # Set environment variable for the collection name
            env = os.environ.copy()
            env["COLLECTION_NAME"] = collection_name

            result = subprocess.run(
                [
                    "python",
                    "/app/scripts/warm_start.py",
                    "--ef", ef,
                    "--limit", limit
                ],
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            print(f"Successfully warmed {collection_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error warming {collection_name}: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            sys.exit(1)

    print("All collections warmed successfully")

if __name__ == "__main__":
    main()
