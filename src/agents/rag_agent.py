from __future__ import annotations
import logging
from dataclasses import dataclass, field

from src.retrieval.hybrid_retriever import RetrievedDoc

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Tracks the agent's working state across iterations."""
    original_query: str
    queries_used: list[str] = field(default_factory=list)
    all_retrieved: list[RetrievedDoc] = field(default_factory=list)
    final_answer: str = ""
    iterations: int = 0
    confidence: float = 0.0


class RAGAgent:
    """
    Agentic retrieval loop.
    
    Algorithm:
    1. Optionally rewrite query
    2. Optionally generate HyDE hypothetical answer
    3. Optionally expand to N queries
    4. Retrieve for all query variants
    5. Deduplicate and rerank
    6. Assess if context is sufficient (quick LLM call)
    7. If not sufficient and iterations < max: reformulate and repeat
    8. Generate final answer
    """

    def __init__(
        self,
        retriever,
        reranker,
        llm,
        max_iterations: int = 3,
        enable_hyde: bool = True,
        enable_multi_query: bool = True,
        enable_query_rewriting: bool = True,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.max_iterations = max_iterations
        self.enable_hyde = enable_hyde
        self.enable_multi_query = enable_multi_query
        self.enable_query_rewriting = enable_query_rewriting

    def run(self, query: str, top_k: int = 5) -> AgentState:
        """
        Run the full agentic RAG loop.
        
        Returns AgentState with the final answer and all retrieved docs.
        """
        state = AgentState(original_query=query)
        queries = self._build_query_variants(query)

        for iteration in range(self.max_iterations):
            state.iterations += 1
            logger.info(f"Agent iteration {iteration+1}/{self.max_iterations}")

            # Retrieve for all query variants
            all_docs = []
            for q in queries:
                state.queries_used.append(q)
                docs = self.retriever.retrieve(q, top_k=top_k * 2)
                all_docs.extend(docs)

            # Deduplicate by chunk_id
            seen = set()
            deduped = []
            for doc in all_docs:
                if doc.chunk_id not in seen:
                    seen.add(doc.chunk_id)
                    deduped.append(doc)

            # Rerank
            reranked = self.reranker.rerank(query, deduped, top_k=top_k)
            state.all_retrieved = reranked

            # Check if context is sufficient
            if self._is_sufficient(query, reranked) or iteration == self.max_iterations - 1:
                break

            # Not sufficient → reformulate for next iteration
            logger.info("Context insufficient — reformulating query")
            queries = [self.llm.rewrite_query(
                f"The query '{query}' didn't return good results. "
                f"Reformulate it to find different information."
            )]

        # Generate final answer
        state.final_answer = self.llm.answer(query, state.all_retrieved)
        logger.info(f"Agent completed in {state.iterations} iteration(s)")
        return state

    def _build_query_variants(self, query: str) -> list[str]:
        """Build list of queries to run in parallel."""
        queries = [query]

        if self.enable_query_rewriting:
            try:
                rewritten = self.llm.rewrite_query(query)
                if rewritten and rewritten != query:
                    queries.append(rewritten)
                    logger.debug(f"Query rewritten: {rewritten}")
            except Exception as e:
                logger.warning(f"Query rewriting failed: {e}")

        if self.enable_hyde:
            try:
                hyp_doc = self.llm.generate_hypothetical_answer(query)
                queries.append(hyp_doc)
                logger.debug(f"HyDE document generated ({len(hyp_doc)} chars)")
            except Exception as e:
                logger.warning(f"HyDE generation failed: {e}")

        if self.enable_multi_query:
            try:
                expanded = self.llm.expand_query(query, n=2)
                for q in expanded[1:]:  # Skip original (already added)
                    if q not in queries:
                        queries.append(q)
            except Exception as e:
                logger.warning(f"Multi-query expansion failed: {e}")

        logger.info(f"Running {len(queries)} query variant(s)")
        return queries

    def _is_sufficient(self, query: str, docs: list[RetrievedDoc]) -> bool:
        """Quick check: does the retrieved context answer the query?"""
        if not docs:
            return False
        # Heuristic: if top doc has high RRF score, probably sufficient
        if docs and docs[0].rrf_score > 0.02:
            return True
        return False
