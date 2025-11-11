#!/usr/bin/env python3
"""
Build and persist a FAISS index from precomputed training embeddings.
"""

import numpy as np
from services.vectorstore import FAISSStore

EMBEDDINGS_PATH = "src/data/train_split_embeddings.npz"
INDEX_OUTPUT_PATH = "src/data/local_index.faiss"


def main() -> None:
    """Load embeddings, build FAISS index, and save it to disk."""
    data = np.load(EMBEDDINGS_PATH)
    if 'train' not in data:
        raise KeyError(f"'train' array not found in {EMBEDDINGS_PATH}")

    embeddings = data['train']
    store = FAISSStore(max_query_neighbors=5)

    print(f"[INFO] Building FAISS index for {embeddings.shape[0]} embeddings...")
    store.ingest_data(embeddings)
    store.save_index(INDEX_OUTPUT_PATH)
    print(f"[INFO] Index successfully saved to {INDEX_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
