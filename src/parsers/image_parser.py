"""
src/parsers/image_parser.py — Parse images via OCR + Claude vision.

Strategy:
  1. Try pytesseract OCR for text-heavy images (charts, screenshots, scans)
  2. If OCR text is sparse, describe via Claude vision (multimodal)
  3. Return structured ParsedChunk
"""
from __future__ import annotations
import base64
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Reuse ParsedChunk from pdf_parser (or define here for independence)
from src.parsers.pdf_parser import ParsedChunk


class ImageParser:
    """
    Parse image files into text using OCR and/or vision LLM.
    Supports: PNG, JPEG, TIFF, BMP, WebP
    """

    OCR_MIN_CHARS = 30  # If OCR returns fewer than this, use vision LLM instead

    def __init__(self, llm_client=None, prefer_vision: bool = False):
        self.llm_client = llm_client
        self.prefer_vision = prefer_vision

    def parse(self, path: str | Path) -> list[ParsedChunk]:
        path = Path(path)
        logger.info(f"Parsing image: {path.name}")

        # Load image bytes
        with open(path, "rb") as f:
            image_bytes = f.read()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        ext = path.suffix.lstrip(".").lower()
        media_type = self._ext_to_media_type(ext)

        chunks = []

        # ── OCR ───────────────────────────────────────────────────────────
        if not self.prefer_vision:
            ocr_text = self._run_ocr(path)
            if ocr_text and len(ocr_text) >= self.OCR_MIN_CHARS:
                chunks.append(ParsedChunk(
                    content=ocr_text,
                    content_type="text",
                    source=str(path),
                    metadata={"filename": path.name, "method": "ocr"}
                ))
                return chunks

        # ── Vision LLM fallback ───────────────────────────────────────────
        if self.llm_client:
            description = self.llm_client.describe_image(b64, media_type)
            chunks.append(ParsedChunk(
                content=f"[IMAGE DESCRIPTION] {description}",
                content_type="image_description",
                source=str(path),
                metadata={"filename": path.name, "method": "vision_llm"}
            ))
        else:
            logger.warning(f"No LLM client — skipping vision description for {path.name}")

        return chunks

    def _run_ocr(self, path: Path) -> str:
        """Run pytesseract OCR on the image."""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(path)
            text = pytesseract.image_to_string(img, config="--psm 6")
            return text.strip()
        except ImportError:
            logger.warning("pytesseract not installed — skipping OCR")
            return ""
        except Exception as e:
            logger.warning(f"OCR failed for {path}: {e}")
            return ""

    @staticmethod
    def _ext_to_media_type(ext: str) -> str:
        mapping = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
            "tiff": "image/tiff",
            "bmp": "image/bmp",
        }
        return mapping.get(ext, "image/jpeg")
