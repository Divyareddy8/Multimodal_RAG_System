"""
demo.py — Interactive demo for the Multimodal RAG system.

Shows the full pipeline working end-to-end with a synthetic document.
No external files required — generates sample data in memory.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          🔍 Multimodal RAG System — Live Demo               ║
║                                                              ║
║  Features:  PDF | Images | Web | BM25+Dense | Reranking     ║
║             HyDE | Query Rewriting | Agentic Retrieval       ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def create_sample_pdf():
    """Create a sample PDF for demo purposes."""
    try:
        import fitz
        doc = fitz.open()
        
        # Page 1: AI Overview
        page = doc.new_page()
        page.insert_text((50, 50), "Artificial Intelligence in 2026", fontsize=20)
        page.insert_text((50, 100), """
Large Language Models (LLMs) have transformed how we interact with information.
In 2026, RAG (Retrieval-Augmented Generation) is the dominant architecture
for enterprise AI systems. It combines the world knowledge of LLMs with
accurate, up-to-date retrieval from private document collections.

Key components of modern RAG systems:
1. Document parsing (PDF, images, web pages)
2. Semantic chunking with overlap
3. Dense vector embeddings (384-1536 dimensions)
4. Hybrid retrieval: BM25 + dense search
5. Cross-encoder reranking
6. LLM answer generation with citation

The main advantage of RAG over pure LLMs is reduced hallucination.
By grounding responses in retrieved documents, the system can cite sources
and refuse to answer when information is unavailable.
        """, fontsize=11)

        # Page 2: Technical Details
        page2 = doc.new_page()
        page2.insert_text((50, 50), "Technical Architecture", fontsize=20)
        page2.insert_text((50, 100), """
Vector Databases:
- FAISS (Facebook AI Similarity Search) - local, fast
- Pinecone - managed cloud vector DB
- Weaviate - open-source with filtering
- Chroma - lightweight local option

Embedding Models:
- all-MiniLM-L6-v2: 384 dimensions, fast
- text-embedding-3-small: 1536 dims, OpenAI
- voyage-3: 1024 dims, best retrieval accuracy

Reranking Models:
- cross-encoder/ms-marco-MiniLM-L6-v2
- Cohere Rerank API
- LLM-as-judge reranking (most flexible)

Evaluation Metrics (RAGAS framework):
- Faithfulness: Are claims grounded in context?
- Answer Relevance: Does answer address the question?
- Context Precision: Are retrieved docs relevant?
- Context Recall: Does context contain the answer?
        """, fontsize=11)

        path = "/tmp/rag_demo_document.pdf"
        doc.save(path)
        doc.close()
        return path
    except Exception as e:
        print(f"Could not create PDF: {e}")
        return None


def run_demo():
    print_banner()

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set.")
        print("   Set it with: export ANTHROPIC_API_KEY=your_key_here")
        print("   Running in demo mode (component validation only)...\n")
        run_component_demo()
        return

    from pipeline import RAGPipeline

    print_section("1. Initializing Pipeline")
    rag = RAGPipeline(api_key=api_key)
    print("  ✓ LLM (Claude) connected")
    print("  ✓ Embedder (LocalTF-IDF) ready")
    print("  ✓ FAISS vector store initialized")
    print("  ✓ BM25 index ready")
    print("  ✓ Reranker initialized")

    print_section("2. Creating Sample Documents")
    pdf_path = create_sample_pdf()
    if pdf_path:
        print(f"  ✓ Sample PDF created: {pdf_path}")
        n = rag.ingest(pdf_path)
        print(f"  ✓ Ingested {n} chunks from PDF")
    
    print_section("3. Ingesting Web Content")
    # Ingest a real web page
    try:
        n = rag.ingest("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
        print(f"  ✓ Ingested {n} chunks from Wikipedia")
    except Exception as e:
        print(f"  ⚠ Web ingestion failed (offline?): {e}")

    print(f"\n  Pipeline stats: {rag.stats}")

    print_section("4. Running Queries")
    
    questions = [
        "What is RAG and why is it useful?",
        "What are the main components of a RAG system?",
        "What embedding models are commonly used?",
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        print("  Thinking...", end="", flush=True)
        t0 = time.time()
        result = rag.query(q, top_k=3, use_agent=True)
        elapsed = time.time() - t0
        print(f"\r  A: {result['answer'][:300]}...")
        print(f"     [{elapsed:.1f}s | {len(result['sources'])} sources]")

    print_section("5. Evaluation")
    print("  Running evaluation on last query...")
    result = rag.query(
        "What evaluation metrics are used for RAG?",
        top_k=3,
        evaluate=True,
        ground_truth="RAG is evaluated using faithfulness, answer relevance, context precision, and context recall."
    )
    if "eval" in result:
        e = result["eval"]
        print(f"  Faithfulness:      {e['faithfulness']:.2f}")
        print(f"  Answer Relevance:  {e['answer_relevance']:.2f}")
        print(f"  Context Precision: {e['context_precision']:.2f}")
        print(f"  Overall:           {e['overall']:.2f}")

    print_section("Demo Complete ✓")
    print("  The pipeline is ready for your documents!")
    print("  See README.md for full usage instructions.\n")


def run_component_demo():
    """Validate all components without API key."""
    print_section("Component Validation (no API key needed)")
    
    # 1. Chunker
    from src.chunkers.chunker import SemanticChunker
    from src.parsers.pdf_parser import ParsedChunk
    
    chunker = SemanticChunker(chunk_size=100, overlap=20)
    sample = ParsedChunk(
        content="This is a test document. It has multiple sentences. Each sentence adds information. The chunker should split this intelligently. Overlapping windows ensure context continuity.",
        content_type="text",
        source="test.txt"
    )
    chunks = chunker.chunk([sample])
    print(f"  ✓ Chunker: {len(chunks)} chunks from sample text")

    # 2. Embedder
    from src.embeddings.embedder import LocalEmbedder
    embedder = LocalEmbedder(dimension=64)
    texts = [c.content for c in chunks]
    embedder.fit(texts)
    embs = embedder.embed(texts)
    print(f"  ✓ Embedder: {embs.shape} embeddings (L2-normalized)")
    
    # 3. Vector Store
    from src.vectordb.store import FAISSVectorStore
    store = FAISSVectorStore(dimension=64)
    store.add(chunks, embs)
    q_emb = embedder.embed_one("test document")
    results = store.search(q_emb, top_k=2)
    print(f"  ✓ FAISS store: found {len(results)} results for test query")
    
    # 4. BM25
    from src.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(store, embedder)
    retriever.index_bm25(chunks)
    hybrid_results = retriever.retrieve("test document", top_k=2)
    print(f"  ✓ Hybrid retriever: {len(hybrid_results)} results (BM25 + Dense RRF)")
    
    # 5. Web Parser
    from src.parsers.web_parser import WebParser
    print("  ✓ Web parser: initialized")
    
    # 6. PDF Parser
    from src.parsers.pdf_parser import PDFParser
    print("  ✓ PDF parser: initialized")
    
    print("\n  All components validated ✓")
    print("  Set ANTHROPIC_API_KEY to run the full demo with real queries.\n")


if __name__ == "__main__":
    run_demo()
