"""Discovery question and review endpoints."""

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from scientis.agent import get_runner
from scientis.agent.state import WorkflowConfig
from scientis.api.deps import get_db
from scientis.models.db_models import DiscoverySessionRecord
from scientis.models.question import (
    ExportRequest,
    HypothesisGenerateRequest,
    QuestionRequest,
    QuestionResponse,
    ReviewRequest,
)

router = APIRouter()


@router.post("/questions", response_model=QuestionResponse, status_code=202)
async def ask_question(
    req: QuestionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Submit a scientific discovery question.

    Returns immediately with status 'running'. Poll GET /results/{session_id}
    to check for completion.
    """
    session_id = req.session_id or f"sess-{uuid.uuid4().hex[:12]}"

    record = DiscoverySessionRecord(
        session_id=session_id,
        question=req.question,
        status="running",
    )
    db.add(record)
    await db.commit()

    runner = get_runner()
    runner.config = WorkflowConfig(
        max_hypotheses=req.max_hypotheses,
        require_human_review=req.require_review,
    )

    async def _run_discovery() -> None:
        from scientis.db import get_session_factory

        try:
            result = await runner.run(req.question, session_id)
            status = result.get("status", "completed")
            hypotheses = result.get("ranked_hypotheses", [])
            report = result.get("final_report")
            evidence_count = result.get("evidence_count", 0)
        except Exception as e:
            status, hypotheses, report, evidence_count = "error", [], None, 0

        async with get_session_factory()() as session:
            await session.execute(
                update(DiscoverySessionRecord)
                .where(DiscoverySessionRecord.session_id == session_id)
                .values(
                    status=status,
                    hypotheses_json=hypotheses,
                    report=report,
                    evidence_count=evidence_count,
                )
            )
            await session.commit()

    background_tasks.add_task(_run_discovery)

    return QuestionResponse(
        session_id=session_id,
        question=req.question,
        status="running",
    )


@router.get("/results/{session_id}", response_model=QuestionResponse)
async def get_result(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Poll for the status and results of a discovery session."""
    record = await db.get(DiscoverySessionRecord, session_id)
    if not record:
        raise HTTPException(404, f"Session {session_id} not found")
    return _record_to_response(record)


@router.post("/reviews", response_model=QuestionResponse)
async def submit_review(
    req: ReviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit human review decisions and resume the paused workflow."""
    record = await db.get(DiscoverySessionRecord, req.session_id)
    if not record:
        raise HTTPException(404, f"Session {req.session_id} not found")

    runner = get_runner()
    new_state = await runner.resume_with_review(
        req.session_id, req.decisions, req.reviewer
    )

    await db.execute(
        update(DiscoverySessionRecord)
        .where(DiscoverySessionRecord.session_id == req.session_id)
        .values(
            status=new_state.get("status", "completed"),
            hypotheses_json=new_state.get("ranked_hypotheses", record.hypotheses_json),
            report=new_state.get("final_report", record.report),
        )
    )
    await db.commit()
    await db.refresh(record)
    return _record_to_response(record)


@router.post("/hypotheses/generate", response_model=QuestionResponse, status_code=202)
async def generate_hypotheses(
    req: HypothesisGenerateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Generate hypotheses from ingested papers by asking a discovery question."""
    return await ask_question(
        QuestionRequest(question=req.question, session_id=req.session_id),
        background_tasks,
        db,
    )


def _record_to_response(record: DiscoverySessionRecord) -> QuestionResponse:
    return QuestionResponse(
        session_id=record.session_id,
        question=record.question,
        status=record.status,
        evidence_count=record.evidence_count or 0,
        hypotheses=record.hypotheses_json or [],
        report=record.report,
        created_at=record.created_at,
    )
