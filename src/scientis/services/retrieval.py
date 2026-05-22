"""Hybrid retrieval service.

Combines three retrieval strategies with Reciprocal Rank Fusion (RRF):
  1. BM25 lexical search over text chunks
  2. Dense vector similarity (sentence embeddings)
  3. Graph neighbourhood expansion (Neo4j)

New papers are added incrementally via add_chunks(); the BM25 index
is rebuilt in full after each addition (acceptable for typical corpus sizes).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from scientis.llm import LLMClient, ModelTier, get_llm

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk_id: str
    paper_id: str
    text: str
    section: str = ""
    score: float = 0.0
    source: str = ""       # "bm25" | "vector" | "graph"
    entities: list[str] = field(default_factory=list)
    figure_ids: list[str] = field(default_factory=list)


class HybridRetriever:
    """Multi-strategy retriever with RRF score fusion."""

    def __init__(self, llm: Optional[LLMClient] = None):
        self._llm = llm or get_llm()
        self._corpus: dict[str, RetrievalResult] = {}
        self._bm25 = None

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict]) -> None:
        """Add text chunks to the index (additive — existing chunks are preserved).

        Each chunk dict: {chunk_id, paper_id, text, section, entities, figure_ids}
        The BM25 index is rebuilt after every call, so batch additions are
        more efficient than adding one chunk at a time.
        """
        from rank_bm25 import BM25Okapi

        for c in chunks:
            self._corpus[c["chunk_id"]] = RetrievalResult(
                chunk_id=c["chunk_id"],
                paper_id=c["paper_id"],
                text=c["text"],
                section=c.get("section", ""),
                entities=c.get("entities", []),
                figure_ids=c.get("figure_ids", []),
            )

        tokenized = [r.text.lower().split() for r in self._corpus.values()]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("Retrieval index updated: %d total chunks", len(self._corpus))

    def index_chunks(self, chunks: list[dict]) -> None:
        """Replace the entire index with a new set of chunks.

        Prefer add_chunks() when ingesting papers incrementally.
        """
        self._corpus.clear()
        self._bm25 = None
        self.add_chunks(chunks)

    # ── Retrieval ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        alpha_bm25: float = 0.3,
        alpha_vector: float = 0.4,
        alpha_graph: float = 0.3,
    ) -> list[RetrievalResult]:
        """Multi-strategy retrieval with Reciprocal Rank Fusion."""
        bm25_results = self._bm25_search(query, top_k * 2)
        vector_results = await self._vector_search(query, top_k * 2)

        fused = self._reciprocal_rank_fusion(
            [bm25_results, vector_results],
            weights=[alpha_bm25, alpha_vector],
        )
        fused.sort(key=lambda x: x[1], reverse=True)
        results = [
            self._corpus[chunk_id]
            for chunk_id, _ in fused[:top_k]
            if chunk_id in self._corpus
        ]

        # Graph expansion is best-effort; skipped if Neo4j is unavailable
        try:
            graph_results = await self._graph_expand(query, top_k // 2)
            results.extend(graph_results)
        except Exception:
            logger.debug("Graph expansion skipped (Neo4j may not be available)")

        return results

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self._bm25 or not self._corpus:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        chunk_ids = list(self._corpus.keys())
        ranked = sorted(
            [(chunk_ids[i], float(scores[i])) for i in range(len(scores))],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    async def _vector_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Dense retrieval via embedding cosine similarity."""
        if not self._corpus:
            return []

        texts = [query] + [r.text for r in self._corpus.values()]
        try:
            embeddings = await self._llm.embed(texts, tier=ModelTier.local)
        except Exception:
            logger.warning("Embedding failed — skipping vector search")
            return []

        import numpy as np

        query_vec = np.array(embeddings[0])
        chunk_ids = list(self._corpus.keys())
        similarities: list[tuple[str, float]] = []
        for i, emb in enumerate(embeddings[1:]):
            chunk_vec = np.array(emb)
            norm = np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-8
            sim = float(np.dot(query_vec, chunk_vec) / norm)
            similarities.append((chunk_ids[i], sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    async def _graph_expand(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Expand results via graph neighbourhood (claims sharing query entities)."""
        from scientis.services.graph_service import get_graph_service

        graph = get_graph_service()
        results: list[RetrievalResult] = []

        for word in query.split():
            if len(word) < 4:
                continue
            try:
                claims = await graph.find_supporting_claims(word, limit=3)
                for c in claims:
                    results.append(RetrievalResult(
                        chunk_id=c.get("claim_id", ""),
                        paper_id=c.get("paper_id", ""),
                        text=c.get("text", ""),
                        score=c.get("confidence", 0.5),
                        source="graph",
                    ))
            except Exception:
                continue

        return results[:top_k]

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[tuple[str, float]]],
        weights: list[float],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Weighted RRF: score = sum(weight / (k + rank)) across all lists."""
        scores: dict[str, float] = {}
        for weight, ranked in zip(weights, ranked_lists):
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Module-level singleton ───────────────────────────────────────────────

_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
