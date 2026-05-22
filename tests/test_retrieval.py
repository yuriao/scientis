"""Unit tests for the hybrid retriever."""

import pytest

from scientis.services.retrieval import HybridRetriever


def _make_chunks(paper_id: str, n: int) -> list[dict]:
    return [
        {
            "chunk_id": f"{paper_id}-c{i}",
            "paper_id": paper_id,
            "text": f"Sample text number {i} about mitochondria and neurodegeneration.",
            "section": f"page_{i}",
            "entities": ["mitochondria"],
            "figure_ids": [],
        }
        for i in range(n)
    ]


def test_add_chunks_is_additive():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._corpus = {}
    retriever._bm25 = None
    retriever._llm = None

    retriever.add_chunks(_make_chunks("paper-1", 3))
    assert len(retriever._corpus) == 3

    retriever.add_chunks(_make_chunks("paper-2", 2))
    assert len(retriever._corpus) == 5


def test_index_chunks_replaces_corpus():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._corpus = {}
    retriever._bm25 = None
    retriever._llm = None

    retriever.add_chunks(_make_chunks("paper-1", 5))
    retriever.index_chunks(_make_chunks("paper-2", 2))  # full replacement
    assert len(retriever._corpus) == 2
    assert all("paper-2" in cid for cid in retriever._corpus)


def test_bm25_search_returns_ranked_results():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._corpus = {}
    retriever._bm25 = None
    retriever._llm = None

    retriever.add_chunks(_make_chunks("p1", 4))
    results = retriever._bm25_search("mitochondria neurodegeneration", top_k=2)
    assert len(results) == 2
    # Scores should be in descending order
    assert results[0][1] >= results[1][1]


def test_rrf_merges_lists():
    merged = HybridRetriever._reciprocal_rank_fusion(
        [["a", "b", "c"], ["b", "a", "d"]],  # type: ignore[arg-type]
        weights=[0.5, 0.5],
    )
    # 'b' and 'a' appear in both lists so should score higher than 'c' or 'd'
    ids = [item[0] for item in merged]
    assert ids.index("b") < ids.index("c")
    assert ids.index("a") < ids.index("d")
