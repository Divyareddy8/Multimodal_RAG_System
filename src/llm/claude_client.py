from __future__ import annotations
import json
import logging
import os
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)


class ClaudeClient:
    """
    Thin wrapper around the Anthropic API for RAG use cases.
    
    Model: claude-sonnet-4-6 (cost-effective, long context)
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_MAX_TOKENS = 2048

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Export it or pass api_key= to ClaudeClient."
            )
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

    # ──────────────────────────────────────────────────────────────────────
    # Core completion
    # ──────────────────────────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.1,
    ) -> str:
        """Simple single-turn completion."""
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def chat(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Multi-turn chat completion."""
        kwargs = dict(model=self.model, max_tokens=max_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    # ──────────────────────────────────────────────────────────────────────
    # RAG Answer Generation
    # ──────────────────────────────────────────────────────────────────────

    RAG_SYSTEM = """\
You are a precise, factual assistant. Answer questions using ONLY the provided context.
If the context doesn't contain the answer, say "I don't have enough information to answer this."
Always cite your sources by mentioning the document name and page when available.
"""

    def answer(self, query: str, context_docs: list, max_tokens: int = 1024) -> str:
        """
        Generate a grounded answer from retrieved context.
        
        Args:
            query: user question
            context_docs: list of RetrievedDoc objects
            
        Returns:
            Answer string
        """
        context = self._format_context(context_docs)
        prompt = f"""Context documents:
{context}

Question: {query}

Answer based strictly on the context above:"""

        return self.complete(prompt, system=self.RAG_SYSTEM, max_tokens=max_tokens)

    def _format_context(self, docs: list) -> str:
        """Format retrieved docs into a context string."""
        parts = []
        for i, doc in enumerate(docs, 1):
            source_info = f"[Doc {i} | {doc.source}"
            if doc.page:
                source_info += f" | Page {doc.page}"
            source_info += "]"
            parts.append(f"{source_info}\n{doc.content}")
        return "\n\n---\n\n".join(parts)

    # ──────────────────────────────────────────────────────────────────────
    # Query Rewriting
    # ──────────────────────────────────────────────────────────────────────

    def rewrite_query(self, query: str) -> str:
        """Expand/clarify the query for better retrieval."""
        prompt = f"""Rewrite the following search query to be more specific and retrieve better results.
Keep it concise. Return ONLY the rewritten query, nothing else.

Original query: {query}
Rewritten query:"""
        return self.complete(prompt, max_tokens=100).strip()

    def generate_hypothetical_answer(self, query: str) -> str:
        """
        HyDE: Generate a hypothetical answer to improve dense retrieval.
        The embedding of a plausible answer is closer to relevant docs
        than the embedding of the question itself.
        """
        prompt = f"""Write a brief, plausible answer to the following question.
This is used for search purposes — be factual and concise (2-4 sentences).
Question: {query}
Hypothetical answer:"""
        return self.complete(prompt, max_tokens=150).strip()

    def expand_query(self, query: str, n: int = 3) -> list[str]:
        """
        Generate N alternative phrasings of the query for multi-query retrieval.
        Returns list of query strings including the original.
        """
        prompt = f"""Generate {n} different ways to ask the following question.
Return ONLY a JSON array of strings, no explanation.

Question: {query}
Alternative phrasings:"""
        try:
            response = self.complete(prompt, max_tokens=200).strip()
            alternatives = json.loads(response)
            return [query] + alternatives[:n]
        except Exception:
            return [query]

    # ──────────────────────────────────────────────────────────────────────
    # Vision
    # ──────────────────────────────────────────────────────────────────────

    def describe_image(self, base64_data: str, media_type: str = "image/jpeg") -> str:
        """Describe an image using Claude's vision capability."""
        # Normalize media_type
        if not media_type.startswith("image/"):
            media_type = f"image/{media_type}"

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail for a search index. "
                        "Include: main subject, text visible, data/charts if present, "
                        "colors, layout, and any key information."
                    ),
                },
            ],
        }]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=messages,
        )
        return response.content[0].text
