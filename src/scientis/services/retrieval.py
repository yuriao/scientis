"""Hybrid retrieval service.

Combines three retrieval strategies with reciprocal rank fusion:
  1. BM25 lexical search over text chunks
  2. Dense vector similarity (embeddings)
  3. Graph neighborhood expansion (Neo4j)

Returns ranked, provenance-tracked results.
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
    source: str = ""  # "bm25", "vector", "graph"
    entities: list[str] = field(default_factory=list)
    figure_ids: list[str] = field(default_factory=list)


class HybridRetriever:
    """Multi-strategy retriever with fusion ranking."""

    def __init__(self, llm: Optional[LLMClient] = None):
        self._llm = llm or get_llm()
        self._corpus: dict[str, RetrievalResult] = {}
        self._bm25 = None

    # ── Indexing ───────────────────────────────────────

    def index_chunks(self, chunks: list[dict]) -> None:
        """Index text chunks for retrieval.

        Each chunk: {chunk_id, paper_id, text, section, entities, figure_ids}
        """
        from rank_bm25 import BM25Okapi

        self._corpus = {}
        corpus_texts = []

        for c in chunks:
            chunk_id = c["chunk_id"]
            self._corpus[chunk_id] = RetrievalResult(
                chunk_id=chunk_id,
                paper_id=c["paper_id"],
                text=c["text"],
                section=c.get("section", ""),
                entities=c.get("entities", []),
                figure_ids=c.get("figure_ids", []),
            )
            corpus_texts.append(c["text"])

        # Tokenize for BM25
        tokenized = [text.lower().split() for text in corpus_texts]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("Indexed %d chunks for hybrid retrieval", len(chunks))

    # ── Retrieval ──────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        alpha_bm25: float = 0.3,
        alpha_vector: float = 0.4,
        alpha_graph: float = 0.3,
    ) -> list[RetrievalResult]:
        """Multi-strategy retrieval with reciprocal rank fusion."""
        bm25_results = self._bm25_search(query, top_k * 2)
        vector_results = await self._vector_search(query, top_k * 2)

        # Fuse scores
        fused = self._reciprocal_rank_fusion(
            [bm25_results, vector_results],
            weights=[alpha_bm25, alpha_vector],
            k=60,
        )
        # Sort and trim
        fused.sort(key=lambda x: x[1], reverse=True)
        results = [self._corpus[chunk_id] for chunk_id, _ in fused[:top_k]]

        # If we have graph service, expand with neighbors
        try:
            graph_results = await self._graph_expand(query, top_k // 2)
            results.extend(graph_results)
        except Exception:
            logger.debug("Graph expansion skipped (Neo4j may not be available)")

        return results

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self._bm25 or not self._corpus:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        chunk_ids = list(self._corpus.keys())
        ranked = sorted(
            [(chunk_ids[i], float(scores[i])) for i in range(len(scores))],
            key=lambda x: x[1], reverse=True,
        )
        return ranked[:top_k]

    async def _vector_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Dense retrieval via embedding similarity."""
        if not self._corpus:
            return []

        # Embed query and all chunks
        texts = [query] + [r.text for r in self._corpus.values()]
        try:
            embeddings = await self._llm.embed(texts, tier=ModelTier.local)
        except Exception:
            logger.warning("Embedding failed, skipping vector search")
            return []

        query_emb = embeddings[0]
        chunk_embs = embeddings[1:]

        # Cosine similarity
        import numpy as np
        query_vec = np.array(query_emb)
        chunk_ids = list(self._corpus.keys())

        similarities = []
        for i, emb in enumerate(chunk_embs):
            chunk_vec = np.array(emb)
            sim = float(np.dot(query_vec, chunk_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-8
            ))
            similarities.append((chunk_ids[i], sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    async def _graph_expand(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Expand results via graph neighborhood (shared entities/mechanisms)."""
        from scientis.services.graph_service import get_graph_service

        graph = get_graph_service()

        # Extract potential entity names from query (simple approach)
        # In production, use NER or the LLM to extract entities
        words = query.split()
        results = []

        for word in words:
            if len(word) < 3:
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
        """Combine multiple ranked lists using weighted RRF."""
        scores: dict[str, float] = {}
        for weight, ranked in zip(weights, ranked_lists):
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
