"""Scientific understanding service.

Extracts structured claims, entities, and relations from parsed papers.
Uses tiered LLM: cheap for figure captions, local/heavy for claim extraction.
"""

import json
import logging
import uuid
from typing import Optional

from scientis.llm import LLMClient, ModelTier, get_llm
from scientis.llm.schemas import CLAIM_EXTRACTION_SCHEMA, ENTITY_CANONICALIZATION_SCHEMA
from scientis.models.claim import Claim, EvidenceSpan
from scientis.storage.object_store import ObjectStore

logger = logging.getLogger(__name__)


async def extract_claims(
    paper_id: str,
    store: ObjectStore,
    llm: Optional[LLMClient] = None,
) -> list[Claim]:
    """Extract structured claims from a parsed paper.

    1. Load paper_text.json from object store
    2. Chunk text by section, send to LLM with structured output
    3. Parse response, create Claim objects with evidence spans
    """
    if llm is None:
        llm = get_llm()

    # Load paper text
    text_key = f"papers/{paper_id}/artifacts/paper_text.json"
    text_data = json.loads(store.get(text_key))
    pages = text_data.get("pages", [])

    # Build full text by section
    full_text = "\n\n".join(p["text"] for p in pages)

    # Split into manageable chunks (~6000 chars each)
    chunks = _chunk_text(full_text, max_chars=6000)

    all_claims: list[Claim] = []
    for i, chunk in enumerate(chunks[:10]):  # Cap at 10 chunks to control cost
        try:
            resp = await llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a scientific evidence extraction system. "
                            "Extract ALL claims from the paper section below. "
                            "A claim must be a declarative, falsifiable statement. "
                            "For each claim, provide evidence spans (verbatim quotes), "
                            "entities mentioned, and confidence based on evidence quality. "
                            "Include contradictory text if the paper itself notes limitations."
                        ),
                    },
                    {"role": "user", "content": f"Paper text section:\n\n{chunk}"},
                ],
                tier=ModelTier.cheap,
                response_format=CLAIM_EXTRACTION_SCHEMA,
                max_tokens=4096,
                temperature=0.1,
            )

            data = json.loads(resp.content)
            for c in data.get("claims", []):
                evidence_spans = []
                for es in c.get("evidence_spans", []):
                    evidence_spans.append(EvidenceSpan(
                        type=es.get("type", "text"),
                        quote=es.get("quote", ""),
                        figure_id=es.get("figure_id", ""),
                        panel=es.get("panel", ""),
                    ))

                claim_id = f"c-{uuid.uuid4().hex[:12]}"
                all_claims.append(Claim(
                    claim_id=claim_id,
                    paper_id=paper_id,
                    claim=c["claim"],
                    section=c.get("section", ""),
                    evidence=evidence_spans,
                    entities=c.get("entities", []),
                    confidence=c.get("confidence", 0.5),
                    contradicting_evidence=[
                        EvidenceSpan(type="text", quote=c["contradicting_text"])
                    ] if c.get("contradicting_text") else [],
                ))

        except Exception:
            logger.exception("Claim extraction failed for chunk %d of %s", i, paper_id)
            continue

    logger.info("Extracted %d claims from paper %s (%d chunks)", len(all_claims), paper_id, len(chunks))
    return all_claims


async def canonicalize_entity(
    entity_name: str,
    entity_type: str,
    llm: Optional[LLMClient] = None,
) -> dict:
    """Canonicalize an entity name using LLM (resolve synonyms)."""
    if llm is None:
        llm = get_llm()

    resp = await llm.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a biomedical entity normalization system. "
                    "Given an entity name and type, return the canonical name, "
                    "known aliases, and your confidence."
                ),
            },
            {
                "role": "user",
                "content": f"Entity: {entity_name}\nType: {entity_type}",
            },
        ],
        tier=ModelTier.cheap,
        response_format=ENTITY_CANONICALIZATION_SCHEMA,
        max_tokens=512,
        temperature=0.0,
    )
    return json.loads(resp.content)


def _chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current.strip():
        chunks.append(current.strip())
    return chunks
