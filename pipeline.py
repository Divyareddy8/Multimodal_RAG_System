from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.config import RAGConfig, DEFAULT_CONFIG
from src.parsers.pdf_parser import PDFParser
from src.parsers.image_parser import ImageParser
from src.parsers.web_parser import WebParser
from src.chunkers.chunker import SemanticChunker
from src.embeddings.embedder import get_embedder
from src.vectordb.store import FAISSVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import LLMReranker
from src.llm.claude_client import ClaudeClient
from src.agents.rag_agent import RAGAgent
from src.evaluation.evaluator import RAGEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end Multimodal RAG Pipeline.
    
    Supports ingestion of:
      - PDF files
      - Image files (PNG, JPEG, etc.)
      - Web URLs
    
    And querying with:
      - Hybrid retrieval (BM25 + Dense)
      - Cross-encoder reranking
      - Query rewriting + HyDE
      - Agentic multi-step retrieval
    """

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG, api_key: Optional[str] = None):
        self.config = config
        logger.info("Initializing RAG Pipeline...")

        # ── LLM ───────────────────────────────────────────────────────────
        self.llm = ClaudeClient(
            api_key=api_key or config.llm.api_key,
            model=config.llm.model,
        )

        # ── Parsers ───────────────────────────────────────────────────────
        self.pdf_parser = PDFParser(describe_images=False)  # Set True to use vision
        self.image_parser = ImageParser(llm_client=self.llm)
        self.web_parser = WebParser()

        # ── Chunker ───────────────────────────────────────────────────────
        self.chunker = SemanticChunker(
            chunk_size=config.chunking.chunk_size,
            overlap=config.chunking.chunk_overlap,
            min_size=config.chunking.min_chunk_size,
        )

        # ── Embedder ──────────────────────────────────────────────────────
        self.embedder = get_embedder(
            model=config.embedding.model,
            dimension=config.embedding.dimension,
        )

        # ── Vector Store ──────────────────────────────────────────────────
        self.vector_store = FAISSVectorStore(dimension=config.embedding.dimension)

        # ── Retriever (initialized after first ingest) ────────────────────
        self.retriever: Optional[HybridRetriever] = None
        self._all_chunks: list = []

        # ── Reranker ──────────────────────────────────────────────────────
        self.reranker = LLMReranker(self.llm)

        # ── Evaluator ─────────────────────────────────────────────────────
        self.evaluator = RAGEvaluator(self.llm)

        logger.info("RAG Pipeline ready ✓")

    # ──────────────────────────────────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────────────────────────────────

    def ingest(self, source: Union[str, Path]) -> int:
        """
        Ingest a document into the pipeline.
        
        Args:
            source: file path (PDF/image) or URL
            
        Returns:
            Number of chunks added
        """
        source_str = str(source)
        logger.info(f"Ingesting: {source_str}")

        # ── Route to appropriate parser ───────────────────────────────────
        if source_str.startswith("http://") or source_str.startswith("https://"):
            parsed = self.web_parser.parse(source_str)
        else:
            path = Path(source_str)
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                parsed = self.pdf_parser.parse(path)
            elif suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".bmp"}:
                parsed = self.image_parser.parse(path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

        if not parsed:
            logger.warning(f"No content extracted from {source_str}")
            return 0

        # ── Chunk ─────────────────────────────────────────────────────────
        chunks = self.chunker.chunk(parsed)
        self._all_chunks.extend(chunks)

        # ── Embed ─────────────────────────────────────────────────────────
        texts = [c.content for c in chunks]
        self.embedder.fit(texts)  # Updates vocab incrementally
        embeddings = self.embedder.embed(texts)

        # ── Store ─────────────────────────────────────────────────────────
        self.vector_store.add(chunks, embeddings)

        # ── Rebuild retriever with new chunks ─────────────────────────────
        self._rebuild_retriever()

        logger.info(f"Ingested {len(chunks)} chunks from {source_str}")
        return len(chunks)

    def ingest_batch(self, sources: list) -> dict:
        """Ingest multiple sources."""
        results = {}
        for source in sources:
            try:
                n = self.ingest(source)
                results[str(source)] = {"status": "ok", "chunks": n}
            except Exception as e:
                logger.error(f"Failed to ingest {source}: {e}")
                results[str(source)] = {"status": "error", "error": str(e)}
        return results

    def _rebuild_retriever(self):
        """Rebuild HybridRetriever with all current chunks."""
        # Re-fit embedder on full corpus
        all_texts = [c.content for c in self._all_chunks]
        self.embedder.fit(all_texts)

        # Rebuild FAISS index with fresh embeddings
        self.vector_store = FAISSVectorStore(dimension=self.config.embedding.dimension)
        embeddings = self.embedder.embed(all_texts)
        self.vector_store.add(self._all_chunks, embeddings)

        # Rebuild retriever
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            rrf_k=self.config.retrieval.rrf_k,
        )
        self.retriever.index_bm25(self._all_chunks)
        logger.debug(f"Retriever rebuilt with {len(self._all_chunks)} chunks")

    # ──────────────────────────────────────────────────────────────────────
    # Querying
    # ──────────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_agent: bool = True,
        evaluate: bool = False,
        ground_truth: Optional[str] = None,
    ) -> dict:
        """
        Query the RAG pipeline.
        
        Args:
            question: user question
            top_k: number of docs to retrieve
            use_agent: use agentic retrieval loop (recommended)
            evaluate: run evaluation metrics
            ground_truth: for evaluation (optional)
            
        Returns:
            dict with 'answer', 'sources', optional 'eval'
        """
        if self.retriever is None or self.vector_store.is_empty:
            return {
                "answer": "No documents ingested yet. Please call ingest() first.",
                "sources": [],
            }

        logger.info(f"Query: {question[:80]}...")

        if use_agent:
            agent = RAGAgent(
                retriever=self.retriever,
                reranker=self.reranker,
                llm=self.llm,
                max_iterations=self.config.agent.max_iterations,
                enable_hyde=self.config.agent.enable_hyde,
                enable_multi_query=self.config.agent.enable_multi_query,
                enable_query_rewriting=self.config.agent.enable_query_rewriting,
            )
            state = agent.run(question, top_k=top_k)
            answer = state.final_answer
            docs = state.all_retrieved
        else:
            # Simple retrieval without agent
            docs = self.retriever.retrieve(question, top_k=top_k)
            docs = self.reranker.rerank(question, docs, top_k=top_k)
            answer = self.llm.answer(question, docs)

        result = {
            "answer": answer,
            "sources": [
                {
                    "source": d.source,
                    "page": d.page,
                    "content_type": d.content_type,
                    "score": round(d.rrf_score, 4),
                    "preview": d.content[:200] + "..." if len(d.content) > 200 else d.content,
                }
                for d in docs
            ],
        }

        if evaluate:
            eval_result = self.evaluator.evaluate(
                query=question,
                answer=answer,
                context_docs=docs,
                ground_truth=ground_truth,
            )
            result["eval"] = {
                "faithfulness": eval_result.faithfulness,
                "answer_relevance": eval_result.answer_relevance,
                "context_precision": eval_result.context_precision,
                "context_recall": eval_result.context_recall,
                "overall": eval_result.overall,
            }

        return result

    def save(self, directory: str = "./data/vector_store"):
        """Persist the vector store to disk."""
        self.vector_store.save(directory)

    def load(self, directory: str = "./data/vector_store"):
        """Load a previously saved vector store."""
        self.vector_store.load(directory)
        self._rebuild_retriever()

    @property
    def stats(self) -> dict:
        return {
            "total_chunks": len(self._all_chunks),
            "vector_store_size": len(self.vector_store),
            "embedding_vocab": len(self.embedder._vocab) if hasattr(self.embedder, "_vocab") else "N/A",
        }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Multimodal RAG Pipeline")
    parser.add_argument("--ingest", nargs="+", help="Files or URLs to ingest")
    parser.add_argument("--query", type=str, help="Question to ask")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--save", type=str, default=None, help="Save vector store to path")
    parser.add_argument("--load", type=str, default=None, help="Load vector store from path")
    args = parser.parse_args()

    rag = RAGPipeline()

    if args.load:
        rag.load(args.load)
        print(f"Loaded vector store from {args.load}")

    if args.ingest:
        for source in args.ingest:
            n = rag.ingest(source)
            print(f"Ingested {n} chunks from {source}")

    if args.query:
        result = rag.query(args.query, evaluate=args.eval)
        print(f"\n{'='*60}")
        print(f"Question: {args.query}")
        print(f"{'='*60}")
        print(f"Answer:\n{result['answer']}")
        print(f"\nSources ({len(result['sources'])}):")
        for s in result["sources"]:
            print(f"  - {s['source']} (page {s['page']}, score {s['score']})")
        if "eval" in result:
            print(f"\nEvaluation:")
            for k, v in result["eval"].items():
                print(f"  {k}: {v:.2f}")

    if args.save:
        rag.save(args.save)
        print(f"Saved to {args.save}")

    print(f"\nPipeline stats: {rag.stats}")
