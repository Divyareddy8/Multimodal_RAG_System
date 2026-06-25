# 🔍 Multimodal RAG System

A production-grade Retrieval-Augmented Generation pipeline supporting PDFs, images, tables, and web pages.

## Architecture

```
Input Sources (PDF / Image / Table / Web)
         ↓
   Document Parser         ← pymupdf, pytesseract, requests
         ↓
  Semantic Chunker         ← sliding window + sentence boundaries
         ↓
  Embedding Model          ← local sentence-transformers or OpenAI
         ↓
    Vector Store           ← FAISS (local) or Chroma (persistent)
         ↓
  Hybrid Retriever         ← BM25 + Dense (RRF fusion)
         ↓
   Cross-Encoder           ← Reranking pass
         ↓
  Query Rewriter           ← HyDE / multi-query expansion
         ↓
 Agentic Retrieval         ← tool-calling loop
         ↓
   LLM (Claude)            ← final answer generation
         ↓
  Eval Framework           ← faithfulness, relevance, recall
```

## Project Structure

```
multimodal-rag/
├── src/
│   ├── parsers/           # Document ingestion
│   │   ├── pdf_parser.py
│   │   ├── image_parser.py
│   │   ├── table_parser.py
│   │   └── web_parser.py
│   ├── chunkers/          # Text splitting strategies
│   │   └── chunker.py
│   ├── embeddings/        # Embedding models
│   │   └── embedder.py
│   ├── vectordb/          # Vector store abstraction
│   │   └── store.py
│   ├── retrieval/         # Hybrid retrieval + reranking
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── llm/               # LLM integration
│   │   └── claude_client.py
│   ├── agents/            # Agentic retrieval loop
│   │   └── rag_agent.py
│   └── evaluation/        # Eval framework
│       └── evaluator.py
├── configs/
│   └── config.py
├── pipeline.py            # Main RAG pipeline
├── demo.py                # Interactive demo
└── requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here

# 3. Run the demo
python demo.py

# 4. Ingest documents and query
python pipeline.py
```

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| PDF Parsing | ✅ | Text + images + tables from PDFs |
| Image OCR | ✅ | pytesseract + Claude vision |
| Web Scraping | ✅ | HTML → clean text |
| Semantic Chunking | ✅ | Sliding window with overlap |
| Dense Retrieval | ✅ | FAISS vector search |
| BM25 Retrieval | ✅ | Keyword-based sparse retrieval |
| Hybrid Fusion | ✅ | Reciprocal Rank Fusion (RRF) |
| Reranking | ✅ | Cross-encoder reranking pass |
| Query Rewriting | ✅ | HyDE + multi-query expansion |
| Agentic Retrieval | ✅ | LLM-driven tool-use loop |
| Evaluation | ✅ | Faithfulness + relevance + recall |
