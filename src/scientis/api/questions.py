"""Question and discovery endpoints."""

from fastapi import APIRouter, BackgroundTasks, HTTPException

from scientis.agent import get_runner
from scientis.agent.state import WorkflowConfig
from scientis.models.question import (
    HypothesisGenerateRequest,
    QuestionRequest,
    QuestionResponse,
    ReviewRequest,
)

router = APIRouter()

# In-memory result store (replace with PostgreSQL in production)
_results: dict[str, QuestionResponse] = {}


@router.post("/questions", response_model=QuestionResponse, status_code=202)
async def ask_question(req: QuestionRequest, background_tasks: BackgroundTasks):
    """Submit a scientific discovery question.

    Example: "What shared mechanisms explain AD, PD, ALS, and FTD?"
    """
    runner = get_runner()
    config = WorkflowConfig(
        max_hypotheses=req.max_hypotheses,
        require_human_review=req.require_review,
    )
    runner.config = config

    # Run in background
    async def _run_discovery():
        try:
            result = await runner.run(req.question, req.session_id)
            _results[result.get("session_id", "")] = QuestionResponse(
                session_id=result.get("session_id", ""),
                question=req.question,
                status=result.get("status", "error"),
                evidence_count=result.get("evidence_count", 0),
                hypotheses=result.get("ranked_hypotheses", []),
                report=result.get("final_report"),
            )
        except Exception as e:
            _results[req.session_id or ""] = QuestionResponse(
                session_id=req.session_id or "",
                question=req.question,
                status="error",
            )

    background_tasks.add_task(_run_discovery)

    session_id = req.session_id or f"sess-{id(req)}"
    response = QuestionResponse(
        session_id=session_id,
        question=req.question,
        status="running",
    )
    _results[session_id] = response
    return response


@router.get("/results/{session_id}", response_model=QuestionResponse)
async def get_result(session_id: str):
    """Get the status and results of a discovery session."""
    result = _results.get(session_id)
    if not result:
        raise HTTPException(404, f"Session {session_id} not found")
    return result


@router.post("/reviews", response_model=QuestionResponse)
async def submit_review(req: ReviewRequest):
    """Submit human review decisions and resume the workflow."""
    result = _results.get(req.session_id)
    if not result:
        raise HTTPException(404, f"Session {req.session_id} not found")

    runner = get_runner()
    new_state = await runner.resume_with_review(
        req.session_id, req.decisions, req.reviewer
    )

    result.status = new_state.get("status", "completed")
    result.hypotheses = new_state.get("ranked_hypotheses", result.hypotheses)
    result.report = new_state.get("final_report", result.report)
    _results[req.session_id] = result

    return result


@router.post("/hypotheses/generate", response_model=QuestionResponse, status_code=202)
async def generate_hypotheses(req: HypothesisGenerateRequest, background_tasks: BackgroundTasks):
    """Generate hypotheses from a subset of papers."""
    # Reuse the question flow with optional paper filter
    return await ask_question(
        QuestionRequest(question=req.question, session_id=req.session_id),
        background_tasks,
    )
