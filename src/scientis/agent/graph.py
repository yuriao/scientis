"""LangGraph workflow definition for scientific discovery.

Workflow:
  expand_query → retrieve_context → compile_evidence →
  induce_mechanism → check_contradictions → human_review → publish
"""

import logging

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from scientis.agent.state import AgentState
from scientis.agent.tools import (
    check_contradictions,
    compile_evidence,
    expand_query,
    induce_mechanism,
    retrieve_context,
)

logger = logging.getLogger(__name__)


async def human_review(state: AgentState) -> dict:
    """Human-in-the-loop review gate.

    When novel, high-confidence hypotheses are found, execution pauses here
    and waits for the caller to provide decisions via Command(resume=...).
    If review is not required, the node passes through immediately.
    """
    hypotheses = state.get("ranked_hypotheses", [])
    weakened = state.get("weakened_hypotheses", [])
    config = state.get("config", {})
    require_review = (
        config.get("require_human_review", True) if isinstance(config, dict) else True
    )

    novel_found = any(
        h.get("confidence", 0) > 0.5 and h.get("hypothesis_id") not in weakened
        for h in hypotheses
    )

    if require_review and novel_found:
        # Pause execution. Caller resumes with:
        #   Command(resume={"decisions": [...], "reviewer": "..."})
        payload = interrupt({
            "type": "review_required",
            "hypotheses": hypotheses,
            "message": "Review these hypotheses before publication.",
        })
        decisions = payload.get("decisions", []) if isinstance(payload, dict) else []
        reviewer = payload.get("reviewer", "") if isinstance(payload, dict) else ""
        return {
            "review_decisions": decisions,
            "reviewed_by": reviewer,
            "review_required": True,
            "status": "completed",
        }

    return {"review_required": False, "status": "completed"}


async def publish(state: AgentState) -> dict:
    """Compile the final markdown report from hypotheses and counterevidence."""
    hypotheses = state.get("ranked_hypotheses", [])
    counterevidence = state.get("counterevidence_findings", [])

    parts = [
        "# Scientific Discovery Report\n",
        f"## Question\n{state.get('question', '')}\n",
        f"## Evidence Retrieved\n{state.get('evidence_count', 0)} chunks across papers\n",
    ]

    if hypotheses:
        parts.append("## Mechanistic Hypotheses\n")
        for i, h in enumerate(hypotheses, 1):
            parts.append(
                f"### {i}. {h.get('mechanism', 'Unknown mechanism')}\n"
                f"**Confidence:** {h.get('confidence', 0):.0%}\n\n"
                f"{h.get('description', '')}\n\n"
                f"**Supporting claims:** {len(h.get('supporting_claims', []))}\n"
                f"**Contradicting claims:** {len(h.get('contradicting_claims', []))}\n"
                f"**Next experiments:** {', '.join(h.get('next_experiments', []))}\n\n"
            )

    if counterevidence:
        parts.append("## Counterevidence\n")
        for ce in counterevidence:
            for weakness in ce.get("weaknesses", []):
                parts.append(f"- [{ce.get('severity', '?')}] {weakness}\n")

    return {"final_report": "".join(parts), "status": "completed"}


def build_workflow() -> StateGraph:
    """Build and return the uncompiled scientific discovery workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("expand_query", expand_query)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("compile_evidence", compile_evidence)
    workflow.add_node("induce_mechanism", induce_mechanism)
    workflow.add_node("check_contradictions", check_contradictions)
    workflow.add_node("human_review", human_review)
    workflow.add_node("publish", publish)

    workflow.set_entry_point("expand_query")
    workflow.add_edge("expand_query", "retrieve_context")
    workflow.add_edge("retrieve_context", "compile_evidence")
    workflow.add_edge("compile_evidence", "induce_mechanism")
    workflow.add_edge("induce_mechanism", "check_contradictions")
    workflow.add_edge("check_contradictions", "human_review")
    workflow.add_edge("human_review", "publish")
    workflow.add_edge("publish", END)

    return workflow
