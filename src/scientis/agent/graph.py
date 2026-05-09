"""LangGraph workflow definition.

Workflow:
  question → expand_query → retrieve_context → compile_evidence →
  induce_mechanism → check_contradictions → human_review → publish
"""

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from scientis.agent.state import AgentState
from scientis.agent.tools import (
    check_contradictions,
    compile_evidence,
    expand_query,
    human_review_gate,
    induce_mechanism,
    retrieve_context,
)

logger = logging.getLogger(__name__)


def should_review(state: AgentState) -> Literal["await_review", "publish"]:
    """Route after human_review_gate."""
    if state.get("review_required"):
        return "await_review"
    return "publish"


def after_review(state: AgentState) -> Literal["publish", "revise"]:
    """Route after human provides review decisions."""
    decisions = state.get("review_decisions", [])
    if any(d.get("action") == "revise" for d in decisions):
        return "revise"
    return "publish"


async def publish(state: AgentState) -> AgentState:
    """Compile final report and mark complete."""
    hypotheses = state.get("ranked_hypotheses", [])
    counterevidence = state.get("counterevidence_findings", [])

    report_parts = [
        f"# Scientific Discovery Report\n",
        f"## Question\n{state['question']}\n",
        f"## Evidence Retrieved\n{state.get('evidence_count', 0)} chunks across papers\n",
    ]

    if hypotheses:
        report_parts.append("## Mechanistic Hypotheses\n")
        for i, h in enumerate(hypotheses, 1):
            report_parts.append(
                f"### {i}. {h.get('mechanism', 'Unknown mechanism')}\n"
                f"**Confidence:** {h.get('confidence', 0):.0%}\n\n"
                f"{h.get('description', '')}\n\n"
                f"**Supporting claims:** {len(h.get('supporting_claims', []))}\n"
                f"**Contradicting claims:** {len(h.get('contradicting_claims', []))}\n"
                f"**Next experiments:** {', '.join(h.get('next_experiments', []))}\n\n"
            )

    if counterevidence:
        report_parts.append("## Counterevidence\n")
        for ce in counterevidence:
            for w in ce.get("weaknesses", []):
                report_parts.append(f"- [{ce.get('severity', '?')}] {w}\n")

    report = "".join(report_parts)

    return {
        "final_report": report,
        "status": "completed",
    }


async def handle_error(state: AgentState) -> AgentState:
    """Error recovery node."""
    logger.error("Workflow error: %s", state.get("error_message"))
    return {"status": "error"}


def build_workflow() -> StateGraph:
    """Build the scientific discovery LangGraph workflow."""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("expand_query", expand_query)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("compile_evidence", compile_evidence)
    workflow.add_node("induce_mechanism", induce_mechanism)
    workflow.add_node("check_contradictions", check_contradictions)
    workflow.add_node("human_review", human_review_gate)
    workflow.add_node("publish", publish)
    workflow.add_node("handle_error", handle_error)

    # Set entry point
    workflow.set_entry_point("expand_query")

    # Define edges
    workflow.add_edge("expand_query", "retrieve_context")
    workflow.add_edge("retrieve_context", "compile_evidence")
    workflow.add_edge("compile_evidence", "induce_mechanism")
    workflow.add_edge("induce_mechanism", "check_contradictions")
    workflow.add_edge("check_contradictions", "human_review")

    # Conditional routing after review
    workflow.add_conditional_edges(
        "human_review",
        should_review,
        {
            "await_review": END,  # Pauses — resume when human provides decisions
            "publish": "publish",
        },
    )

    workflow.add_edge("publish", END)
    workflow.add_edge("handle_error", END)

    return workflow


# ── Compiled workflow ─────────────────────────────
_discovery_workflow: StateGraph | None = None


def get_workflow() -> StateGraph:
    global _discovery_workflow
    if _discovery_workflow is None:
        _discovery_workflow = build_workflow().compile()
    return _discovery_workflow
