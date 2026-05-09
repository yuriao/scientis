"""Agent state schema for LangGraph workflow.

The state flows through:
  question → query_expansion → retrieval → evidence_compilation →
  mechanism_induction → counterevidence → human_review → publish
"""

from dataclasses import dataclass, field
from typing import Optional, TypedDict

from scientis.models.hypothesis import Hypothesis
from scientis.services.retrieval import RetrievalResult


class AgentState(TypedDict, total=False):
    """Typed state for the LangGraph agent workflow."""

    # Input
    question: str
    session_id: str

    # Query expansion
    expanded_queries: list[str]
    disease_synonyms: list[str]
    mechanism_synonyms: list[str]

    # Retrieval
    retrieved_chunks: list[dict]
    retrieved_claims: list[dict]
    evidence_count: int

    # Evidence compilation
    comparison_set: list[dict]  # cross-paper evidence matrix
    supporting_papers: list[str]
    conflicting_papers: list[str]

    # Mechanism induction
    hypotheses: list[dict]
    ranked_hypotheses: list[dict]

    # Counterevidence
    counterevidence_findings: list[dict]
    weakened_hypotheses: list[str]

    # Human review
    review_required: bool
    review_decisions: list[dict]  # accept/reject/revise per hypothesis
    reviewed_by: str

    # Output
    status: str  # "running" | "awaiting_review" | "completed" | "error"
    final_report: Optional[str]
    error_message: str

    # Metadata
    iteration_count: int
    model_tier_used: str


@dataclass
class WorkflowConfig:
    """Configuration for the agent workflow."""
    max_retrieval_chunks: int = 50
    max_hypotheses: int = 5
    confidence_threshold: float = 0.3
    require_human_review: bool = True
    model_tier: str = "cheap"  # cheap | local | heavy
