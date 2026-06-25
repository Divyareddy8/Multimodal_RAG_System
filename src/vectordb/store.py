"""
src/vectordb/store.py — FAISS-backed vector store with persistence.

Features:
  - Add / search / delete chunks
  - Save and load from disk
  - Metadata filtering support
  - Returns ranked results with scores
"""
from __future__ import annotations
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    content_type: str
    source: str
    page: int
    score: float          # cosine similarity (higher = better)
    metadata: dict


class FAISSVectorStore:
    """
    In-memory FAISS index with a parallel metadata store.
    
    Stores:
      - FAISS IndexFlatIP (inner product = cosine on normalized vectors)
      - List of Chunk objects (metadata)
    
    Save/load to disk via pickle (index) + JSON (metadata).
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._index = None
        self._chunks: list = []   # parallel list to FAISS index
        self._init_index()

    def _init_index(self):
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
            logger.debug(f"FAISS IndexFlatIP initialized (dim={self.dimension})")
        except ImportError:
            raise ImportError("Install faiss-cpu: pip install faiss-cpu")

    def add(self, chunks: list, embeddings: np.ndarray):
        """
        Add chunks with their precomputed embeddings.
        
        Args:
            chunks: list of Chunk objects
            embeddings: (N, dimension) float32 array, L2-normalized
        """
        assert len(chunks) == len(embeddings), "Chunks and embeddings must match"
        embeddings = embeddings.astype(np.float32)
        self._index.add(embeddings)
        self._chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} vectors. Total: {len(self._chunks)}")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """
        Find top-k most similar chunks.
        
        Args:
            query_embedding: (dimension,) float32, L2-normalized
            top_k: number of results
            
        Returns:
            List of SearchResult sorted by score descending
        """
        if len(self._chunks) == 0:
            return []

        q = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(SearchResult(
                chunk_id=chunk.id,
                content=chunk.content,
                content_type=chunk.content_type,
                source=chunk.source,
                page=chunk.page,
                score=float(score),
                metadata=chunk.metadata,
            ))

        return results

    def save(self, directory: str):
        """Persist the vector store to disk."""
        import faiss
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        index_path = path / "index.faiss"
        meta_path = path / "metadata.pkl"

        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self._chunks, f)

        logger.info(f"Vector store saved to {directory} ({len(self._chunks)} vectors)")

    def load(self, directory: str):
        """Load a previously saved vector store."""
        import faiss
        path = Path(directory)
        index_path = path / "index.faiss"
        meta_path = path / "metadata.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"No index at {index_path}")

        self._index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self._chunks = pickle.load(f)

        logger.info(f"Vector store loaded from {directory} ({len(self._chunks)} vectors)")

    def __len__(self):
        return len(self._chunks)

    @property
    def is_empty(self):
        return len(self._chunks) == 0
