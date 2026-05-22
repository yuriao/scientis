"""Unit tests for the paper processing pipeline helpers."""

from scientis.services.pipeline import _build_chunks


def test_build_chunks_splits_long_text():
    pages = [{"page_num": 1, "text": "A" * 2500}]
    chunks = _build_chunks("p-test", pages)
    assert len(chunks) == 3  # ceil(2500 / 1000)
    for c in chunks:
        assert c["paper_id"] == "p-test"
        assert len(c["text"]) <= 1000


def test_build_chunks_skips_blank_pages():
    pages = [
        {"page_num": 1, "text": "   "},
        {"page_num": 2, "text": "Some content here."},
    ]
    chunks = _build_chunks("p-test", pages)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Some content here."


def test_build_chunks_unique_ids():
    pages = [{"page_num": i, "text": "word " * 300} for i in range(1, 4)]
    chunks = _build_chunks("p-test", pages)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id values must be unique"
