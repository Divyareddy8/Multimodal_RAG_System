from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    query: str
    answer: str
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    scores: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def overall(self) -> float:
        active = [v for v in [
            self.faithfulness,
            self.answer_relevance,
            self.context_recall,
            self.context_precision
        ] if v > 0]
        return sum(active) / len(active) if active else 0.0

    def summary(self) -> str:
        lines = [
            f"Query: {self.query[:80]}",
            f"{'─'*60}",
            f"  Faithfulness:      {self.faithfulness:.2f}/1.0",
            f"  Answer Relevance:  {self.answer_relevance:.2f}/1.0",
            f"  Context Recall:    {self.context_recall:.2f}/1.0",
            f"  Context Precision: {self.context_precision:.2f}/1.0",
            f"  Overall Score:     {self.overall:.2f}/1.0",
        ]
        if self.errors:
            lines.append(f"  Errors: {', '.join(self.errors)}")
        return "\n".join(lines)


class RAGEvaluator:
    """
    LLM-as-judge evaluation framework for RAG pipelines.
    All metrics scored 0-1 via Claude.
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def evaluate(
        self,
        query: str,
        answer: str,
        context_docs: list,
        ground_truth: Optional[str] = None,
    ) -> EvalResult:
        result = EvalResult(query=query, answer=answer)
        context_str = self._format_context(context_docs)

        # ── Faithfulness ──────────────────────────────────────────────────
        try:
            result.faithfulness = self._score_faithfulness(answer, context_str)
        except Exception as e:
            result.errors.append(f"faithfulness: {e}")

        # ── Answer Relevance ──────────────────────────────────────────────
        try:
            result.answer_relevance = self._score_answer_relevance(query, answer)
        except Exception as e:
            result.errors.append(f"answer_relevance: {e}")

        # ── Context Precision ─────────────────────────────────────────────
        try:
            result.context_precision = self._score_context_precision(query, context_docs)
        except Exception as e:
            result.errors.append(f"context_precision: {e}")

        # ── Context Recall (requires ground truth) ────────────────────────
        if ground_truth:
            try:
                result.context_recall = self._score_context_recall(ground_truth, context_str)
            except Exception as e:
                result.errors.append(f"context_recall: {e}")

        return result

    def evaluate_batch(self, test_cases: list[dict]) -> list[EvalResult]:
        """
        Evaluate multiple test cases.
        
        Each test_case dict: {query, answer, context_docs, ground_truth (optional)}
        """
        results = []
        for i, tc in enumerate(test_cases):
            logger.info(f"Evaluating case {i+1}/{len(test_cases)}")
            result = self.evaluate(**tc)
            results.append(result)
        return results

    def aggregate(self, results: list[EvalResult]) -> dict:
        """Compute mean scores across a batch."""
        if not results:
            return {}
        metrics = ["faithfulness", "answer_relevance", "context_precision", "context_recall", "overall"]
        return {
            m: sum(getattr(r, m) for r in results) / len(results)
            for m in metrics
        }

    # ──────────────────────────────────────────────────────────────────────
    # Individual metric scorers
    # ──────────────────────────────────────────────────────────────────────

    def _score_faithfulness(self, answer: str, context: str) -> float:
        prompt = f"""You are evaluating whether an AI answer is faithful to its source context.

Context:
{context[:2000]}

Answer:
{answer[:1000]}

Score the faithfulness from 0 to 1:
- 1.0: All claims in the answer are directly supported by the context
- 0.5: Some claims are supported, some are not
- 0.0: The answer contradicts or goes beyond the context

Respond ONLY with JSON: {{"score": <0-1>, "reason": "<brief reason>"}}"""
        return self._llm_score(prompt)

    def _score_answer_relevance(self, query: str, answer: str) -> float:
        prompt = f"""Score whether the answer addresses the question.

Question: {query}
Answer: {answer[:1000]}

Score 0-1:
- 1.0: Directly and completely answers the question
- 0.5: Partially answers the question
- 0.0: Does not answer the question

Respond ONLY with JSON: {{"score": <0-1>, "reason": "<brief reason>"}}"""
        return self._llm_score(prompt)

    def _score_context_precision(self, query: str, docs: list) -> float:
        if not docs:
            return 0.0
        relevant_count = 0
        for doc in docs[:5]:  # Evaluate top 5
            prompt = f"""Is this passage relevant to the query?

Query: {query}
Passage: {doc.content[:500]}

Respond ONLY with JSON: {{"relevant": true/false}}"""
            try:
                response = self.llm.complete(prompt, max_tokens=30)
                data = json.loads(response.strip())
                if data.get("relevant", False):
                    relevant_count += 1
            except Exception:
                pass
        return relevant_count / min(len(docs), 5)

    def _score_context_recall(self, ground_truth: str, context: str) -> float:
        prompt = f"""Does the context contain enough information to derive the ground truth answer?

Ground Truth: {ground_truth[:500]}
Context: {context[:2000]}

Score 0-1:
- 1.0: Context fully contains the information needed
- 0.5: Context partially contains the information
- 0.0: Context lacks the necessary information

Respond ONLY with JSON: {{"score": <0-1>, "reason": "<brief reason>"}}"""
        return self._llm_score(prompt)

    def _llm_score(self, prompt: str) -> float:
        """Call LLM and parse a score from JSON response."""
        response = self.llm.complete(prompt, max_tokens=100)
        try:
            data = json.loads(response.strip())
            return float(data.get("score", 0.5))
        except Exception:
            # Try to extract a number from the text
            import re
            match = re.search(r"\d+\.\d+|\d+", response)
            if match:
                val = float(match.group())
                return min(val, 1.0) if val <= 1.0 else val / 10.0
            return 0.5

    @staticmethod
    def _format_context(docs: list) -> str:
        return "\n\n".join(
            f"[{i+1}] {doc.content[:500]}"
            for i, doc in enumerate(docs)
        )
