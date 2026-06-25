from __future__ import annotations
import json
import logging
from typing import Optional

from src.retrieval.hybrid_retriever import RetrievedDoc

logger = logging.getLogger(__name__)


class LLMReranker:
    """
    Use Claude to rerank retrieved documents by query relevance.
    
    Prompt: score each passage 1-10 on relevance to the query.
    Processes in batches of 5 to stay within context limits.
    """

    RERANK_PROMPT = """\
You are a relevance judge. Given a query and a passage, score the passage's relevance to the query on a scale of 1-10.

Query: {query}

Passage:
{passage}

Respond with ONLY a JSON object: {{"score": <1-10>}}
"""

    def __init__(self, llm_client, batch_size: int = 10):
        self.llm = llm_client
        self.batch_size = batch_size

    def rerank(self, query: str, docs: list[RetrievedDoc], top_k: int = 5) -> list[RetrievedDoc]:
        """Rerank docs and return top_k by LLM relevance score."""
        if not docs:
            return docs

        scored = []
        for doc in docs:
            score = self._score(query, doc.content)
            scored.append((score, doc))

        scored.sort(key=lambda x: -x[0])
        result = [doc for _, doc in scored[:top_k]]
        logger.info(f"Reranked {len(docs)} → {len(result)} docs")
        return result

    def _score(self, query: str, passage: str) -> float:
        """Score a single passage against the query."""
        # Truncate passage to avoid token waste
        passage_preview = passage[:800]
        prompt = self.RERANK_PROMPT.format(query=query, passage=passage_preview)

        try:
            response = self.llm.complete(prompt, max_tokens=20)
            # Parse JSON score
            data = json.loads(response.strip())
            return float(data.get("score", 5))
        except Exception as e:
            logger.warning(f"Reranking score parse failed: {e}")
            return 5.0  # Neutral fallback


class CrossEncoderReranker:
    """
    Rerank using a sentence-transformers cross-encoder.
    Much faster than LLM reranking, very accurate.
    
    Install: pip install sentence-transformers
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast)
           cross-encoder/ms-marco-electra-base (accurate)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            logger.info(f"CrossEncoder loaded: {model_name}")
        except ImportError:
            raise ImportError("pip install sentence-transformers")

    def rerank(self, query: str, docs: list[RetrievedDoc], top_k: int = 5) -> list[RetrievedDoc]:
        if not docs:
            return docs
        pairs = [(query, doc.content) for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: -x[0])
        return [doc for _, doc in ranked[:top_k]]


def get_reranker(backend: str = "llm", llm_client=None):
    if backend == "llm":
        if llm_client is None:
            raise ValueError("LLMReranker requires llm_client")
        return LLMReranker(llm_client)
    elif backend == "cross_encoder":
        return CrossEncoderReranker()
    else:
        raise ValueError(f"Unknown reranker: {backend}")
