"""Paper domain models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata captured at ingestion."""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    doi: str = ""
    arxiv_id: str = ""
    year: Optional[int] = None
    journal: str = ""
    abstract: str = ""
    keywords: list[str] = Field(default_factory=list)
    source: str = ""  # "upload", "arxiv", "pmc", etc.


class PaperSummary(BaseModel):
    """Paper record stored in PostgreSQL."""
    paper_id: str = Field(..., description="Canonical paper ID (e.g. p-2026-0509-abc123)")
    filename: str
    checksum: str
    status: str = "ingested"  # ingested | parsing | parsed | understanding | error
    metadata: PaperMetadata = Field(default_factory=PaperMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
