"""Hypothesis models for agentic reasoning."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    """A generated hypothesis from cross-paper reasoning."""
    hypothesis_id: str = Field(..., description="Canonical hypothesis ID (e.g. h-abc123)")
    mechanism: str  # e.g. "mitochondrial dysfunction"
    description: str
    supporting_claims: list[str] = Field(default_factory=list)  # claim_ids
    contradicting_claims: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    diseases: list[str] = Field(default_factory=list)
    genes: list[str] = Field(default_factory=list)
    pathways: list[str] = Field(default_factory=list)
    next_experiments: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_by: Optional[str] = None  # human curator
    status: str = "proposed"  # proposed | approved | rejected | revised
