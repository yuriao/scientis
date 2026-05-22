"""Agent tools for the LangGraph scientific discovery workflow.

Each tool is an async callable that receives the current AgentState dict
and returns a partial state update.
"""

import json
import logging
import uuid
from typing import Any

from scientis.llm import ModelTier, get_llm
from scientis.llm.schemas import MECHANISM_INDUCTION_SCHEMA
from scientis.services.retrieval import get_retriever

logger = logging.getLogger(__name__)


# ── Tool: Expand Query ───────────────────────────────────────────────

async def expand_query(state: dict[str, Any]) -> dict[str, Any]:
    """Expand the user's question with disease synonyms, mechanism synonyms,
    and reformulated search queries to improve retrieval coverage."""
    question = state["question"]
    llm = get_llm()

    resp = await llm.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a biomedical query expansion system. "
                    "Given a scientific question, return: "
                    "1) Disease synonyms (abbreviations, related conditions) "
                    "2) Mechanism synonyms (alternative terms for biological processes) "
                    "3) Reformulated search queries optimised for retrieval.\n"
                    "Output as JSON: {diseases: [...], mechanisms: [...], queries: [...]}"
                ),
            },
            {"role": "user", "content": question},
        ],
        tier=ModelTier.cheap,
        max_tokens=512,
        temperature=0.0,
    )

    try:
        data = json.loads(resp.content)
    except json.JSONDecodeError:
        data = {"diseases": [], "mechanisms": [], "queries": [question]}

    return {
        "disease_synonyms": data.get("diseases", []),
        "mechanism_synonyms": data.get("mechanisms", []),
        "expanded_queries": data.get("queries", [question]),
    }


# ── Tool: Retrieve Context ───────────────────────────────────────────

async def retrieve_context(state: dict[str, Any]) -> dict[str, Any]:
    """Hybrid retrieval (BM25 + vector + graph) across all indexed papers."""
    queries = state.get("expanded_queries", [state["question"]])
    retriever = get_retriever()

    all_chunks: list[dict] = []
    for query in queries[:3]:
        results = await retriever.retrieve(query, top_k=20)
        for r in results:
            all_chunks.append({
                "chunk_id": r.chunk_id,
                "paper_id": r.paper_id,
                "text": r.text,
                "section": r.section,
                "score": r.score,
                "source": r.source,
                "entities": r.entities,
                "figure_ids": r.figure_ids,
            })

    # Deduplicate by chunk_id, keep highest score
    seen: set[str] = set()
    unique: list[dict] = []
    for chunk in all_chunks:
        if chunk["chunk_id"] not in seen:
            seen.add(chunk["chunk_id"])
            unique.append(chunk)

    unique.sort(key=lambda x: x["score"], reverse=True)

    config = state.get("config", {})
    max_chunks = config.get("max_retrieval_chunks", 50) if isinstance(config, dict) else 50
    top_chunks = unique[:max_chunks]

    return {"retrieved_chunks": top_chunks, "evidence_count": len(top_chunks)}


# ── Tool: Compile Evidence ────────────────────────────────────────────

async def compile_evidence(state: dict[str, Any]) -> dict[str, Any]:
    """Build a cross-paper comparison matrix: which papers support vs conflict."""
    chunks = state.get("retrieved_chunks", [])
    question = state["question"]
    llm = get_llm()

    if not chunks:
        return {"comparison_set": [], "supporting_papers": [], "conflicting_papers": []}

    context = "\n\n---\n\n".join(
        f"[{c['paper_id']}] ({c.get('section', '?')}) {c['text'][:500]}"
        for c in chunks[:15]
    )

    resp = await llm.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a scientific evidence compiler. Analyse the paper excerpts "
                    "and classify each paper's position relative to the user's question. "
                    "Output JSON:\n"
                    "{comparison: [{paper_id, position: supports|conflicts|neutral, "
                    "summary, key_claims: [...]}], cross_cutting_themes: []}"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nEvidence:\n{context}",
            },
        ],
        tier=ModelTier.cheap,
        max_tokens=2048,
        temperature=0.1,
    )

    try:
        data = json.loads(resp.content)
        comparison = data.get("comparison", [])
    except json.JSONDecodeError:
        comparison = []

    supporting = [c["paper_id"] for c in comparison if c.get("position") == "supports"]
    conflicting = [c["paper_id"] for c in comparison if c.get("position") == "conflicts"]

    return {
        "comparison_set": comparison,
        "supporting_papers": supporting,
        "conflicting_papers": conflicting,
    }


# ── Tool: Induce Mechanism ───────────────────────────────────────────

async def induce_mechanism(state: dict[str, Any]) -> dict[str, Any]:
    """Cluster evidence into shared mechanistic themes and generate hypotheses."""
    chunks = state.get("retrieved_chunks", [])
    comparison = state.get("comparison_set", [])
    question = state["question"]
    llm = get_llm()

    if not chunks:
        return {"hypotheses": [], "ranked_hypotheses": []}

    context = "\n\n".join(
        f"PAPER {c['paper_id']}: {c['text'][:400]}" for c in chunks[:20]
    )
    comparison_text = json.dumps(comparison[:10], indent=2)

    resp = await llm.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a mechanistic reasoning engine for biomedical discovery. "
                    "Given a question, retrieved evidence, and a comparison matrix, "
                    "induce shared mechanisms that explain the observations. "
                    "For each mechanism, provide supporting and contradicting claims, "
                    "affected diseases/genes/pathways, confidence, next experiments, "
                    "and knowledge gaps. Be rigorous: flag where evidence is weak."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Evidence:\n{context}\n\n"
                    f"Comparison Matrix:\n{comparison_text}"
                ),
            },
        ],
        tier=ModelTier.cheap,
        response_format=MECHANISM_INDUCTION_SCHEMA,
        max_tokens=4096,
        temperature=0.2,
    )

    try:
        data = json.loads(resp.content)
        hypotheses = data.get("hypotheses", [])
    except json.JSONDecodeError:
        hypotheses = []

    for h in hypotheses:
        h["hypothesis_id"] = f"h-{uuid.uuid4().hex[:12]}"

    hypotheses.sort(key=lambda h: h.get("confidence", 0), reverse=True)

    return {
        "hypotheses": hypotheses,
        "ranked_hypotheses": hypotheses[:5],
    }


# ── Tool: Check Contradictions ─────────────────────────────────────────

async def check_contradictions(state: dict[str, Any]) -> dict[str, Any]:
    """Force the agent to find evidence that weakens each ranked hypothesis."""
    hypotheses = state.get("ranked_hypotheses", [])
    chunks = state.get("retrieved_chunks", [])
    llm = get_llm()

    if not hypotheses:
        return {"counterevidence_findings": [], "weakened_hypotheses": []}

    hp_text = json.dumps([
        {
            "id": h.get("hypothesis_id"),
            "mechanism": h.get("mechanism"),
            "claims": h.get("supporting_claims", [])[:5],
        }
        for h in hypotheses
    ])

    context = "\n\n".join(
        f"[{c['paper_id']}] {c['text'][:300]}" for c in chunks[:15]
    )

    resp = await llm.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a scientific sceptic. For each hypothesis, find evidence "
                    "that weakens or contradicts it. Consider: methodological limitations, "
                    "conflicting results, alternative explanations, publication bias, "
                    "sample size issues. Output JSON:\n"
                    "{findings: [{hypothesis_id, weaknesses: [...], severity: low|medium|high}]}"
                ),
            },
            {
                "role": "user",
                "content": f"Hypotheses:\n{hp_text}\n\nEvidence pool:\n{context}",
            },
        ],
        tier=ModelTier.cheap,
        max_tokens=2048,
        temperature=0.1,
    )

    try:
        data = json.loads(resp.content)
        findings = data.get("findings", [])
    except json.JSONDecodeError:
        findings = []

    weakened = [
        f["hypothesis_id"]
        for f in findings
        if f.get("severity") == "high"
    ]

    return {
        "counterevidence_findings": findings,
        "weakened_hypotheses": weakened,
    }
