"""
src/chunkers/chunker.py — Split raw parsed content into retrieval-ready chunks.

Strategy: Sliding window over sentences, respecting token limits.
Each chunk gets a unique ID + full metadata lineage.
"""
from __future__ import annotations
import hashlib
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Final retrieval unit."""
    id: str                          # stable SHA256 hash of content
    content: str
    content_type: str                # "text" | "table" | "image_description"
    source: str
    page: int = 0
    chunk_index: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = self._hash(self.content)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]


class SemanticChunker:
    """
    Splits ParsedChunks into Chunks using a sliding window over sentences.
    
    - Tables and image descriptions: kept as single chunks (no splitting)
    - Text: split by sentence boundaries, grouped to fit token budget
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64, min_size: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_size = min_size

    def chunk(self, parsed_chunks) -> list[Chunk]:
        """
        Input:  list of ParsedChunk (from parsers)
        Output: list of Chunk (ready for embedding)
        """
        all_chunks: list[Chunk] = []
        for pc in parsed_chunks:
            if pc.content_type in ("table", "image_description"):
                # Preserve as-is
                chunk = Chunk(
                    id=Chunk._hash(pc.content),
                    content=pc.content,
                    content_type=pc.content_type,
                    source=pc.source,
                    page=pc.page,
                    chunk_index=0,
                    token_count=self._count_tokens(pc.content),
                    metadata=pc.metadata,
                )
                all_chunks.append(chunk)
            else:
                text_chunks = self._split_text(pc.content)
                for i, text in enumerate(text_chunks):
                    chunk = Chunk(
                        id=Chunk._hash(text + pc.source + str(i)),
                        content=text,
                        content_type="text",
                        source=pc.source,
                        page=pc.page,
                        chunk_index=i,
                        token_count=self._count_tokens(text),
                        metadata=pc.metadata,
                    )
                    all_chunks.append(chunk)

        logger.info(f"Chunker produced {len(all_chunks)} chunks")
        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping sentence-boundary chunks."""
        sentences = self._split_sentences(text)
        chunks = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            s_tokens = self._count_tokens(sentence)
            
            if current_tokens + s_tokens > self.chunk_size and current:
                chunk_text = " ".join(current).strip()
                if len(chunk_text) >= self.min_size:
                    chunks.append(chunk_text)
                # Overlap: keep last N tokens worth of sentences
                current, current_tokens = self._trim_for_overlap(current)

            current.append(sentence)
            current_tokens += s_tokens

        # Final chunk
        if current:
            chunk_text = " ".join(current).strip()
            if len(chunk_text) >= self.min_size:
                chunks.append(chunk_text)

        return chunks or [text[:2000]]  # Fallback

    def _trim_for_overlap(self, sentences: list[str]) -> tuple[list[str], int]:
        """Keep the tail of sentences up to overlap token budget."""
        trimmed = []
        tokens = 0
        for s in reversed(sentences):
            t = self._count_tokens(s)
            if tokens + t > self.overlap:
                break
            trimmed.insert(0, s)
            tokens += t
        return trimmed, tokens

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Simple sentence splitter using regex."""
        # Split on '. ', '! ', '? ' or newlines
        raw = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Fast approximate token count (whitespace-based)."""
        return max(1, len(text.split()))
