"""
configs/config.py — Central configuration for the RAG system.
"""
from dataclasses import dataclass, field
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ChunkingConfig:
    chunk_size: int = 512          # tokens per chunk
    chunk_overlap: int = 64        # overlap between chunks
    min_chunk_size: int = 50       # discard chunks smaller than this


@dataclass
class EmbeddingConfig:
    model: str = "local"           # "local" | "openai" | "voyage"
    dimension: int = 384           # embedding dimension
    batch_size: int = 32


@dataclass
class RetrievalConfig:
    top_k: int = 10                # candidates from vector store
    bm25_weight: float = 0.4       # weight for BM25 in hybrid fusion
    dense_weight: float = 0.6      # weight for dense in hybrid fusion
    rerank_top_k: int = 5          # final docs after reranking
    rrf_k: int = 60                # RRF constant (60 is standard)


@dataclass
class LLMConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048
    temperature: float = 0.1
    context_window: int = 180_000
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )


@dataclass
class AgentConfig:
    max_iterations: int = 5        # max retrieval rounds
    enable_query_rewriting: bool = True
    enable_hyde: bool = True       # Hypothetical Document Embeddings
    enable_multi_query: bool = True


@dataclass
class EvalConfig:
    metrics: list = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevance",
        "context_recall",
        "context_precision",
    ])


@dataclass
class RAGConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    vector_store_path: str = "./data/vector_store"
    log_level: str = "INFO"


# Singleton config
DEFAULT_CONFIG = RAGConfig()
