"""Full paper processing pipeline.

Chains together the processing stages that transform a raw PDF
into indexed, queryable knowledge:

  1. Parse                — extract text and figures from the PDF
  2. Figure Understanding — VLM-based visual analysis of figures and panels
  3. Understand           — extract structured claims via LLM
  4. Graph ingest         — store claims and entity links in Neo4j
  5. Index                — add text chunks to the hybrid retrieval index
"""

import json
import logging
from collections.abc import Callable, Coroutine
from typing import Optional

from scientis.config import Settings
from scientis.services.figure_understanding import process_figure_understanding
from scientis.services.graph_service import get_graph_service
from scientis.services.parsing import parse_paper
from scientis.services.retrieval import get_retriever
from scientis.services.understanding import extract_claims
from scientis.storage.object_store import ObjectStore

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1000  # characters per retrieval chunk

# Optional callback type: async (paper_id, status) -> None
StatusCallback = Optional[Callable[[str, str], Coroutine]]  # noqa: UP045


async def run_pipeline(
    paper_id: str,
    store: ObjectStore,
    settings: Settings,
    on_status: StatusCallback = None,
) -> None:
    """Process a paper through all four pipeline stages.

    on_status, if provided, is awaited after each stage with the new status
    string (e.g. 'parsing', 'understanding', 'indexing', 'ready', 'error').
    """

    async def _update(status: str) -> None:
        if on_status:
            try:
                await on_status(paper_id, status)
            except Exception:
                logger.exception("Status callback failed for paper %s", paper_id)

    try:
        # 1. Parse PDF → text + figures stored in S3
        await _update("parsing")
        parse_result = await parse_paper(paper_id, store, settings)
        logger.info(
            "Parsed %s: %d pages, %d figures",
            paper_id,
            parse_result["pages"],
            parse_result["figures"],
        )

        # 1.5 Figure Understanding (VLM-based visual analysis)
        await _update("figure_understanding")
        fig_result = await process_figure_understanding(paper_id, store)
        logger.info(
            "Figure understanding for %s: %d figures, %d panels, %d unprocessed",
            paper_id,
            sum(len(d.figures) for d in fig_result.figures),
            len(fig_result.panels),
            len(fig_result.unprocessed_figures),
        )

        # 2. Extract structured claims via LLM
        await _update("understanding")
        claims = await extract_claims(paper_id, store)
        logger.info("Extracted %d claims from %s", len(claims), paper_id)

        # 3. Ingest into Neo4j
        graph = get_graph_service()
        await graph.ingest_claims(paper_id, claims)
        for claim in claims:
            if claim.entities:
                await graph.link_claim_entities(claim.claim_id, claim.entities)

        # 4. Build retrieval chunks from the parsed page text
        await _update("indexing")
        text_key = f"papers/{paper_id}/artifacts/paper_text.json"
        text_data = json.loads(store.get(text_key))
        chunks = _build_chunks(paper_id, text_data.get("pages", []))
        get_retriever().add_chunks(chunks)
        logger.info("Indexed %d chunks for %s", len(chunks), paper_id)

        await _update("ready")

    except Exception:
        logger.exception("Pipeline failed for paper %s", paper_id)
        await _update("error")
        raise


def _build_chunks(paper_id: str, pages: list[dict]) -> list[dict]:
    """Split page text into retrieval-sized chunks."""
    chunks = []
    for page in pages:
        text = page.get("text", "")
        page_num = page.get("page_num", 0)
        for i in range(0, len(text), _CHUNK_SIZE):
            chunk_text = text[i : i + _CHUNK_SIZE].strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"{paper_id}-p{page_num}-c{i // _CHUNK_SIZE}",
                        "paper_id": paper_id,
                        "text": chunk_text,
                        "section": f"page_{page_num}",
                        "entities": [],
                        "figure_ids": [],
                    }
                )
    return chunks
