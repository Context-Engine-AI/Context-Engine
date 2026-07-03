#!/usr/bin/env python3
"""
PCA-based projection initialization for CoSQA learning.

Pre-computes PCA on corpus embeddings to initialize the projection layer
with meaningful weights instead of random noise.
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def compute_pca_init_for_collection(
    collection: str,
    sample_limit: int = 1000,
) -> bool:
    """Compute PCA initialization for a collection's projection layer.
    
    Args:
        collection: Qdrant collection name
        sample_limit: Max documents to sample for PCA (for efficiency)
    
    Returns:
        True if successful, False otherwise
    """
    from scripts.embedder import get_embedding_model, get_model_dimension
    from scripts.benchmarks.qdrant_utils import get_qdrant_client
    from scripts.rerank_recursive.projection import LearnedProjection
    
    print(f"Computing PCA initialization for collection: {collection}")
    
    # Get model and dimension
    model = get_embedding_model()
    embed_dim = get_model_dimension()
    print(f"  Embedding model: {os.environ.get('EMBEDDING_MODEL', 'default')}")
    print(f"  Embedding dimension: {embed_dim}")
    
    # Sample documents from collection
    client = get_qdrant_client()
    try:
        info = client.get_collection(collection)
        total_points = info.points_count
        print(f"  Collection has {total_points} points")
    except Exception as e:
        print(f"  ERROR: Failed to get collection info: {e}")
        return False
    
    # Sample points
    sample_size = min(sample_limit, total_points)
    print(f"  Sampling {sample_size} points for PCA...")
    
    try:
        # Scroll through collection to get sample
        points = []
        offset = None
        batch_size = 100
        
        while len(points) < sample_size:
            result = client.scroll(
                collection_name=collection,
                limit=min(batch_size, sample_size - len(points)),
                offset=offset,
                with_payload=True,
                with_vectors=False,  # We'll re-embed from text
            )
            
            batch_points, offset = result
            if not batch_points:
                break
            
            points.extend(batch_points)
            
            if offset is None:
                break
        
        print(f"  Sampled {len(points)} points")
        
    except Exception as e:
        print(f"  ERROR: Failed to sample points: {e}")
        return False
    
    # Extract text and embed
    print(f"  Embedding {len(points)} documents...")
    texts = []
    for point in points:
        payload = point.payload or {}
        # Try to get code or content
        text = payload.get("code") or payload.get("content") or payload.get("text") or ""
        if text:
            texts.append(str(text)[:2000])  # Limit length
    
    if not texts:
        print("  ERROR: No text found in sampled points")
        return False
    
    print(f"  Extracted {len(texts)} text samples")
    
    # Batch embed
    embeddings_list = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embs = list(model.embed(batch))
        embeddings_list.extend(batch_embs)
        if (i // batch_size + 1) % 10 == 0:
            print(f"    Embedded {i+len(batch)}/{len(texts)}")
    
    embeddings = np.array(embeddings_list, dtype=np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # Initialize projection with PCA
    print(f"  Computing PCA projection ({embed_dim} → 256)...")
    projection = LearnedProjection(input_dim=embed_dim, output_dim=256, lr=0.0)
    projection.set_collection(collection)
    projection.init_from_pca(embeddings)
    
    # Save weights
    projection._save_weights()
    print(f"  ✓ Saved PCA-initialized weights to: {projection._weights_path}")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-compute PCA initialization for learning")
    parser.add_argument("--collection", default="cosqa-corpus", help="Collection name")
    parser.add_argument("--sample-limit", type=int, default=1000, help="Max samples for PCA")
    args = parser.parse_args()
    
    success = compute_pca_init_for_collection(
        collection=args.collection,
        sample_limit=args.sample_limit,
    )
    
    sys.exit(0 if success else 1)

