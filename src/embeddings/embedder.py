from __future__ import annotations
import hashlib
import logging
import math
from collections import Counter
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    Lightweight TF-IDF + random projection embedder.
    
    No external models needed — works everywhere.
    For production: swap with SentenceTransformers or OpenAI.
    
    Dimension: configurable (default 384 to match sentence-transformers)
    """

    def __init__(self, dimension: int = 384, seed: int = 42):
        self.dimension = dimension
        self.seed = seed
        self._vocab: dict[str, int] = {}
        self._projection: Optional[np.ndarray] = None
        self._idf: dict[str, float] = {}
        self._is_fitted = False
        self._corpus_size = 0

    def fit(self, texts: list[str]) -> "LocalEmbedder":
        """Build vocabulary and IDF weights from corpus."""
        logger.info(f"Fitting embedder on {len(texts)} texts...")
        self._corpus_size = len(texts)

        # Build vocab
        doc_freq: Counter = Counter()
        tokenized = [self._tokenize(t) for t in texts]
        for tokens in tokenized:
            doc_freq.update(set(tokens))

        # Assign vocab indices
        all_tokens = sorted(doc_freq.keys())
        self._vocab = {tok: i for i, tok in enumerate(all_tokens)}

        # IDF
        N = len(texts)
        self._idf = {
            tok: math.log((N + 1) / (freq + 1)) + 1
            for tok, freq in doc_freq.items()
        }

        # Random projection matrix: vocab_size → dimension
        rng = np.random.RandomState(self.seed)
        vocab_size = len(self._vocab)
        self._projection = rng.randn(vocab_size, self.dimension).astype(np.float32)
        # Normalize columns
        norms = np.linalg.norm(self._projection, axis=0, keepdims=True)
        self._projection /= (norms + 1e-8)

        self._is_fitted = True
        logger.info(f"Embedder fitted. Vocab: {vocab_size}, Dim: {self.dimension}")
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.
        Returns: (N, dimension) float32 array — L2-normalized.
        """
        if not self._is_fitted:
            # Auto-fit on first call
            self.fit(texts)

        embeddings = []
        for text in texts:
            vec = self._tfidf_vector(text)
            projected = vec @ self._projection  # (vocab,) @ (vocab, dim) → (dim,)
            norm = np.linalg.norm(projected)
            if norm > 0:
                projected = projected / norm
            embeddings.append(projected)

        return np.array(embeddings, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _tfidf_vector(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        tf: Counter = Counter(tokens)
        total = sum(tf.values())
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for tok, count in tf.items():
            if tok in self._vocab:
                idx = self._vocab[tok]
                tf_val = count / (total + 1e-8)
                idf_val = self._idf.get(tok, 1.0)
                vec[idx] = tf_val * idf_val
        return vec

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        tokens = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
        return tokens


class SentenceTransformerEmbedder:
    """
    Production embedder using sentence-transformers.
    Install: pip install sentence-transformers
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded SentenceTransformer: {model_name} (dim={self.dimension})")
        except ImportError:
            raise ImportError("pip install sentence-transformers")

    def fit(self, texts):
        return self  # No fitting needed

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def get_embedder(model: str = "local", dimension: int = 384):
    """Factory function."""
    if model == "local":
        return LocalEmbedder(dimension=dimension)
    elif model == "sentence_transformers":
        return SentenceTransformerEmbedder()
    else:
        raise ValueError(f"Unknown embedding model: {model}")
