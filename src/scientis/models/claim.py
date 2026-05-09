"""Claim and evidence models for scientific understanding."""

from typing import Optional

from pydantic import BaseModel, Field


class EvidenceSpan(BaseModel):
    """A span of evidence in the source paper."""
    type: str  # "text" | "figure" | "table"
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    quote: str = ""
    figure_id: str = ""
    panel: str = ""
    explanation: str = ""


class Claim(BaseModel):
    """A structured claim extracted from a paper."""
    claim_id: str = Field(..., description="Canonical claim ID (e.g. c-abc123)")
    paper_id: str
    claim: str  # The claim text
    section: str = ""  # "introduction" | "methods" | "results" | "discussion"
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    relation: str = ""  # "supports" | "refutes" | "neutral"
    entities: list[str] = Field(default_factory=list)  # Genes, proteins, diseases, etc.
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    contradicting_evidence: list[EvidenceSpan] = Field(default_factory=list)
