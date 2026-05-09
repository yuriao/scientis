"""Event models for internal message bus."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PaperUploaded(BaseModel):
    event_type: str = "PaperUploaded"
    paper_id: str
    filename: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaperParsed(BaseModel):
    event_type: str = "PaperParsed"
    paper_id: str
    sections: int = 0
    figures: int = 0
    tables: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ClaimExtracted(BaseModel):
    event_type: str = "ClaimExtracted"
    paper_id: str
    claim_id: str
    evidence_count: int = 0
    confidence: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HypothesisGenerated(BaseModel):
    event_type: str = "HypothesisGenerated"
    hypothesis_id: str
    mechanism: str
    supporting_claims: list[str] = Field(default_factory=list)
    contradicting_claims: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
