"""Paper ingestion and retrieval endpoints."""

import hashlib
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from scientis.api.deps import settings as get_settings
from scientis.config import Settings
from scientis.models.paper import PaperMetadata, PaperSummary
from scientis.services.events import EventBus, PaperUploaded
from scientis.services.ingestion import ingest_paper
from scientis.services.parsing import parse_paper
from scientis.storage.object_store import ObjectStore

router = APIRouter()
_papers_db: dict[str, PaperSummary] = {}  # TODO: replace with PostgreSQL


class PaperUploadResponse(BaseModel):
    paper_id: str
    status: str
    checksum: str
    metadata: PaperMetadata


@router.post("/papers", response_model=PaperUploadResponse, status_code=202)
async def upload_paper(
    file: UploadFile,
    metadata: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    settings: Settings = Depends(get_settings),
):
    """Upload a paper PDF for ingestion and analysis."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    content = await file.read()
    checksum = hashlib.sha256(content).hexdigest()

    # Deduplicate by checksum
    for pid, paper in _papers_db.items():
        if paper.checksum == checksum:
            raise HTTPException(409, f"Duplicate paper (already ingested as {pid})")

    store = ObjectStore(settings)
    paper_id = ingest_paper(store, content, file.filename)

    # Parse metadata if provided
    meta = PaperMetadata()
    if metadata:
        try:
            meta = PaperMetadata.model_validate_json(metadata)
        except Exception:
            pass

    summary = PaperSummary(
        paper_id=paper_id,
        filename=file.filename,
        checksum=checksum,
        status="ingested",
        metadata=meta,
    )
    _papers_db[paper_id] = summary

    # Emit event and enqueue parsing
    EventBus.emit(PaperUploaded(paper_id=paper_id, filename=file.filename))
    background_tasks.add_task(parse_paper, paper_id, store, settings)

    return PaperUploadResponse(
        paper_id=paper_id,
        status="ingested",
        checksum=checksum,
        metadata=meta,
    )


@router.get("/papers/{paper_id}", response_model=PaperSummary)
async def get_paper(paper_id: str):
    """Get paper status and metadata."""
    paper = _papers_db.get(paper_id)
    if not paper:
        raise HTTPException(404, f"Paper {paper_id} not found")
    return paper


@router.get("/papers")
async def list_papers(limit: int = 20, offset: int = 0):
    """List ingested papers."""
    papers = list(_papers_db.values())
    return {
        "total": len(papers),
        "items": papers[offset : offset + limit],
    }
