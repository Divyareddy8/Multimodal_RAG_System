"""
src/parsers/pdf_parser.py — Parse PDFs into structured documents.

Extracts:
  - Text (with page numbers)
  - Tables (as markdown)
  - Images (as base64 for vision LLM)
  - Metadata (title, author, page count)
"""
from __future__ import annotations
import io
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParsedChunk:
    """A raw extracted unit before chunking."""
    content: str
    content_type: str           # "text" | "table" | "image_description"
    source: str                 # file path or URL
    page: int = 0
    metadata: dict = field(default_factory=dict)


class PDFParser:
    """
    Parse PDFs using PyMuPDF (fitz).
    
    Extracts text per page, detects table-like regions,
    and optionally describes embedded images via Claude vision.
    """

    def __init__(self, describe_images: bool = False, llm_client=None):
        self.describe_images = describe_images
        self.llm_client = llm_client  # optional Claude client for image captioning

    def parse(self, path: str | Path) -> list[ParsedChunk]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("Install pymupdf: pip install pymupdf")

        path = Path(path)
        chunks: list[ParsedChunk] = []

        doc = fitz.open(str(path))
        logger.info(f"Parsing PDF: {path.name} ({len(doc)} pages)")

        for page_num, page in enumerate(doc):
            # ── Text ──────────────────────────────────────────────────────
            text = page.get_text("text").strip()
            if text:
                chunks.append(ParsedChunk(
                    content=text,
                    content_type="text",
                    source=str(path),
                    page=page_num + 1,
                    metadata={"filename": path.name}
                ))

            # ── Tables (detected via block layout) ────────────────────────
            tables = self._extract_tables(page)
            for table_md in tables:
                chunks.append(ParsedChunk(
                    content=table_md,
                    content_type="table",
                    source=str(path),
                    page=page_num + 1,
                    metadata={"filename": path.name}
                ))

            # ── Images ────────────────────────────────────────────────────
            if self.describe_images and self.llm_client:
                image_descriptions = self._describe_images(page, doc, path.name, page_num)
                chunks.extend(image_descriptions)

        doc.close()
        logger.info(f"Extracted {len(chunks)} raw chunks from {path.name}")
        return chunks

    def _extract_tables(self, page) -> list[str]:
        """
        Detect table-like structures using PyMuPDF's find_tables().
        Falls back to heuristic if find_tables not available.
        """
        tables = []
        try:
            tab = page.find_tables()
            for t in tab.tables:
                df_data = t.extract()
                if df_data:
                    md = self._list_to_markdown(df_data)
                    tables.append(md)
        except Exception:
            # Older PyMuPDF — skip table detection
            pass
        return tables

    @staticmethod
    def _list_to_markdown(data: list[list]) -> str:
        """Convert a 2D list to a markdown table."""
        if not data:
            return ""
        header = data[0]
        rows = data[1:]
        lines = []
        lines.append("| " + " | ".join(str(c or "") for c in header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c or "") for c in row) + " |")
        return "\n".join(lines)

    def _describe_images(self, page, doc, filename: str, page_num: int) -> list[ParsedChunk]:
        """Extract images from page and describe via Claude vision."""
        chunks = []
        image_list = page.get_images(full=True)
        for img_index, img_ref in enumerate(image_list):
            try:
                xref = img_ref[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                ext = base_image["ext"]
                description = self.llm_client.describe_image(b64, ext)
                chunks.append(ParsedChunk(
                    content=f"[IMAGE] {description}",
                    content_type="image_description",
                    source=filename,
                    page=page_num + 1,
                    metadata={"image_index": img_index}
                ))
            except Exception as e:
                logger.warning(f"Could not process image on page {page_num+1}: {e}")
        return chunks
