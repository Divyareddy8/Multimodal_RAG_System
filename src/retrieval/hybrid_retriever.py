from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.vectordb.store import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    """Unified result format after hybrid fusion."""
    chunk_id: str
    content: str
    content_type: str
    source: str
    page: int
    rrf_score: float          # RRF fusion score
    dense_score: float = 0.0
    bm25_score: float = 0.0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridRetriever:
    """
    Combines dense vector search and BM25 keyword search via RRF.
    
    Usage:
        retriever = HybridRetriever(vector_store, embedder)
        retriever.index_bm25(chunks)        # build BM25 index
        results = retriever.retrieve(query, top_k=5)
    """

    def __init__(self, vector_store, embedder, rrf_k: int = 60):
        self.vector_store = vector_store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_chunks: list = []

    def index_bm25(self, chunks: list):
        """Build BM25 index from chunks."""
        self._bm25_chunks = chunks
        tokenized = [self._tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(chunks)} documents")

    def retrieve(self, query: str, top_k: int = 5, candidates: int = 20) -> list[RetrievedDoc]:
        """
        Hybrid retrieve with RRF fusion.
        
        Args:
            query: user query string
            top_k: final number of results to return
            candidates: how many candidates to pull from each retriever
            
        Returns:
            List of RetrievedDoc sorted by RRF score
        """
        # ── Dense retrieval ───────────────────────────────────────────────
        query_emb = self.embedder.embed_one(query)
        dense_results = self.vector_store.search(query_emb, top_k=candidates)
        dense_ranks = {r.chunk_id: i + 1 for i, r in enumerate(dense_results)}
        dense_scores = {r.chunk_id: r.score for r in dense_results}

        # ── BM25 retrieval ────────────────────────────────────────────────
        bm25_ranks = {}
        bm25_scores_map = {}
        if self._bm25 is not None:
            tokens = self._tokenize(query)
            scores = self._bm25.get_scores(tokens)
            top_indices = np.argsort(scores)[::-1][:candidates]
            for rank, idx in enumerate(top_indices):
                if idx < len(self._bm25_chunks):
                    cid = self._bm25_chunks[idx].id
                    bm25_ranks[cid] = rank + 1
                    bm25_scores_map[cid] = float(scores[idx])

        # ── RRF Fusion ────────────────────────────────────────────────────
        all_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores: dict[str, float] = {}

        for cid in all_ids:
            score = 0.0
            if cid in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[cid])
            if cid in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[cid])
            rrf_scores[cid] = score

        # ── Build result list ─────────────────────────────────────────────
        # Map chunk_id → chunk object
        id_to_dense = {r.chunk_id: r for r in dense_results}
        id_to_bm25_chunk = {}
        if self._bm25_chunks:
            for c in self._bm25_chunks:
                if c.id in bm25_ranks:
                    id_to_bm25_chunk[c.id] = c

        results: list[RetrievedDoc] = []
        for cid, rrf_score in sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]:
            # Get chunk content from whichever retriever found it
            if cid in id_to_dense:
                r = id_to_dense[cid]
                doc = RetrievedDoc(
                    chunk_id=cid,
                    content=r.content,
                    content_type=r.content_type,
                    source=r.source,
                    page=r.page,
                    rrf_score=rrf_score,
                    dense_score=dense_scores.get(cid, 0.0),
                    bm25_score=bm25_scores_map.get(cid, 0.0),
                    metadata=r.metadata,
                )
            elif cid in id_to_bm25_chunk:
                c = id_to_bm25_chunk[cid]
                doc = RetrievedDoc(
                    chunk_id=cid,
                    content=c.content,
                    content_type=c.content_type,
                    source=c.source,
                    page=c.page,
                    rrf_score=rrf_score,
                    dense_score=0.0,
                    bm25_score=bm25_scores_map.get(cid, 0.0),
                    metadata=c.metadata,
                )
            else:
                continue
            results.append(doc)

        logger.debug(f"Hybrid retrieval: {len(results)} results for query '{query[:50]}'")
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        return re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
