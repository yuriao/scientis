"""Paper ingestion and retrieval endpoints."""

import hashlib
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from scientis.api.deps import get_db, get_settings_dep
from scientis.config import Settings
from scientis.models.db_models import PaperRecord
from scientis.models.paper import PaperMetadata, PaperSummary
from scientis.services.graph_service import get_graph_service
from scientis.services.ingestion import ingest_paper
from scientis.services.pipeline import run_pipeline
from scientis.storage.object_store import ObjectStore

router = APIRouter()


@router.post("/papers", response_model=PaperSummary, status_code=202)
async def upload_paper(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    metadata: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings_dep),
):
    """Upload a PDF for ingestion and analysis.

    The file is stored immediately; parsing and claim extraction run
    asynchronously in the background.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    content = await file.read()
    checksum = hashlib.sha256(content).hexdigest()

    # Deduplicate by checksum
    existing = await db.scalar(select(PaperRecord).where(PaperRecord.checksum == checksum))
    if existing:
        raise HTTPException(409, f"Duplicate paper (already ingested as {existing.paper_id})")

    store = ObjectStore(settings)
    paper_id = ingest_paper(store, content, file.filename)

    meta = PaperMetadata()
    if metadata:
        try:
            meta = PaperMetadata.model_validate_json(metadata)
        except Exception:
            pass

    record = PaperRecord(
        paper_id=paper_id,
        filename=file.filename,
        checksum=checksum,
        status="ingested",
        metadata_json=meta.model_dump(),
    )
    db.add(record)
    await db.commit()

    # Store a Paper node in Neo4j (non-blocking failure is acceptable here)
    try:
        summary = PaperSummary(
            paper_id=paper_id,
            filename=file.filename,
            checksum=checksum,
            status="ingested",
            metadata=meta,
        )
        await get_graph_service().create_paper(summary)
    except Exception:
        pass

    async def _status_callback(pid: str, status: str) -> None:
        async with get_db() as session:  # own session for background task
            await session.execute(
                update(PaperRecord)
                .where(PaperRecord.paper_id == pid)
                .values(status=status, updated_at=datetime.utcnow())
            )
            await session.commit()

    background_tasks.add_task(run_pipeline, paper_id, store, settings, _status_callback)

    return PaperSummary(
        paper_id=paper_id,
        filename=file.filename,
        checksum=checksum,
        status="ingested",
        metadata=meta,
    )


@router.get("/papers/{paper_id}", response_model=PaperSummary)
async def get_paper(
    paper_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get paper status and metadata."""
    record = await db.get(PaperRecord, paper_id)
    if not record:
        raise HTTPException(404, f"Paper {paper_id} not found")
    return _record_to_summary(record)


@router.get("/papers")
async def list_papers(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List ingested papers."""
    total_result = await db.execute(select(PaperRecord))
    all_records = total_result.scalars().all()
    page = all_records[offset : offset + limit]
    return {
        "total": len(all_records),
        "items": [_record_to_summary(r) for r in page],
    }


def _record_to_summary(record: PaperRecord) -> PaperSummary:
    return PaperSummary(
        paper_id=record.paper_id,
        filename=record.filename,
        checksum=record.checksum,
        status=record.status,
        metadata=PaperMetadata.model_validate(record.metadata_json or {}),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
