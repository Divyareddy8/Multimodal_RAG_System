"""
src/parsers/web_parser.py — Scrape web pages into clean text chunks.

Pipeline:
  requests → BeautifulSoup → remove boilerplate → extract text + tables
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests

from src.parsers.pdf_parser import ParsedChunk

logger = logging.getLogger(__name__)


class WebParser:
    """
    Fetch and parse web pages into structured text chunks.
    
    Cleans nav/footer boilerplate, extracts <article> or <main> content,
    and captures HTML tables as markdown.
    """

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; RAGBot/1.0; research purposes)"
        )
    }
    TIMEOUT = 15

    def __init__(self, headers: Optional[dict] = None):
        self.headers = headers or self.DEFAULT_HEADERS

    def parse(self, url: str) -> list[ParsedChunk]:
        logger.info(f"Fetching: {url}")
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.TIMEOUT)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return []

        html = resp.text
        return self._html_to_chunks(html, url)

    def _html_to_chunks(self, html: str, source_url: str) -> list[ParsedChunk]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Install beautifulsoup4: pip install beautifulsoup4")

        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "advertisement", "noscript"]):
            tag.decompose()

        chunks = []

        # ── Tables ────────────────────────────────────────────────────────
        for table in soup.find_all("table"):
            md = self._table_to_markdown(table)
            if md:
                chunks.append(ParsedChunk(
                    content=md,
                    content_type="table",
                    source=source_url,
                    metadata={"url": source_url}
                ))
            table.decompose()

        # ── Main content ──────────────────────────────────────────────────
        content_tag = (
            soup.find("article") or
            soup.find("main") or
            soup.find(id=re.compile(r"content|main|article", re.I)) or
            soup.find("body")
        )

        if content_tag:
            text = content_tag.get_text(separator="\n", strip=True)
            text = self._clean_text(text)
            if text:
                chunks.append(ParsedChunk(
                    content=text,
                    content_type="text",
                    source=source_url,
                    metadata={"url": source_url, "title": self._get_title(soup)}
                ))

        logger.info(f"Extracted {len(chunks)} chunks from {source_url}")
        return chunks

    @staticmethod
    def _table_to_markdown(table_tag) -> str:
        rows = table_tag.find_all("tr")
        if not rows:
            return ""
        lines = []
        for i, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            line = "| " + " | ".join(c.get_text(strip=True) for c in cells) + " |"
            lines.append(line)
            if i == 0:
                lines.append("|" + "|".join(["---"] * len(cells)) + "|")
        return "\n".join(lines)

    @staticmethod
    def _clean_text(text: str) -> str:
        # Collapse excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    @staticmethod
    def _get_title(soup) -> str:
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else ""
