import faiss
import numpy as np
import torch
from typing import Optional, Tuple


class FAISSStore:
    """
    A lightweight wrapper for FAISS vector indexing and similarity search.
    Supports L2-normalized embeddings with cosine similarity (via inner product).
    """

    def __init__(self, max_query_neighbors: int = 5):
        """
        Args:
            max_query_neighbors (int): Number of neighbors to retrieve per query.
        """
        self.index: Optional[faiss.Index] = None
        self.max_query_neighbors = max_query_neighbors

    def ingest_data(self, input: np.ndarray) -> None:
        """Ingest and index a matrix of embeddings."""
        if not isinstance(input, np.ndarray):
            raise TypeError("Expected input as np.ndarray")

        # Ensure correct dtype and contiguous memory
        embeddings = np.ascontiguousarray(input, dtype=np.float32)

        faiss.normalize_L2(embeddings)  # in-place normalization

        d = embeddings.shape[1]  # embedding dimension
        index = faiss.IndexFlatIP(d)  # cosine similarity (since normalized)
        self.index = faiss.IndexIDMap(index)

        ids = np.arange(len(embeddings), dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)


    def save_index(self, path: str = "src/data/local_index.faiss") -> None:
        """Persist the current FAISS index to disk."""
        if not self.index:
            raise RuntimeError("No index to save. Call `ingest_data()` first.")
        faiss.write_index(self.index, path)

    def load_index(self, path: str = "src/data/local_index.faiss") -> None:
        """Load an existing FAISS index from disk."""
        self.index = faiss.read_index(path)

    def search_neighbors(
        self, 
        query: np.ndarray | torch.Tensor, 
        metadatas: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform nearest neighbor search.

        Args:
            query (np.ndarray | torch.Tensor): Query embedding (1D or 2D array).
            metadatas (np.ndarray): Array of metadatas aligned with index IDs.

        Returns:
            Tuple of (distances, indices, neighbor_predictions)
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call `ingest_data()` or `load_index()` first.")

        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)

        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, self.max_query_neighbors)

        neighbor_metadatas = metadatas[indices]
        return distances, indices, neighbor_metadatas
