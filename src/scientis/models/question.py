"""Question and result models for the agent API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from scientis.models.hypothesis import Hypothesis


class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    max_hypotheses: int = 5
    require_review: bool = True


class QuestionResponse(BaseModel):
    session_id: str
    question: str
    status: str  # running | awaiting_review | completed | error
    evidence_count: int = 0
    hypotheses: list[dict] = Field(default_factory=list)
    report: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ReviewRequest(BaseModel):
    session_id: str
    reviewer: str = ""
    decisions: list[dict] = Field(
        default_factory=list,
        description="List of {hypothesis_id, action: accept|reject|revise, comment}"
    )


class HypothesisGenerateRequest(BaseModel):
    question: str
    paper_ids: list[str] = Field(default_factory=list)
    session_id: Optional[str] = None


class ExportRequest(BaseModel):
    session_id: str
    format: str = "markdown"  # markdown | slides | json


class ExportResponse(BaseModel):
    session_id: str
    format: str
    content: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
