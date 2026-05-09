"""Export endpoints — reports, slides, JSON."""

from fastapi import APIRouter, HTTPException

from scientis.agent import get_runner
from scientis.models.question import ExportRequest, ExportResponse

router = APIRouter()


@router.post("/exports/report", response_model=ExportResponse)
async def export_report(req: ExportRequest):
    """Export a discovery session as a markdown report."""
    runner = get_runner()

    # Get current state from checkpointer
    try:
        state = await runner._workflow.aget_state(
            {"configurable": {"thread_id": req.session_id}}
        )
        if not state or not state.values:
            raise HTTPException(404, f"Session {req.session_id} not found")

        report = state.values.get("final_report", "")
        if not report:
            # Generate report from current hypotheses
            hypotheses = state.values.get("ranked_hypotheses", [])
            question = state.values.get("question", "")
            report = _build_report(question, hypotheses)

        return ExportResponse(
            session_id=req.session_id,
            format=req.format,
            content=report,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/exports/slides", response_model=ExportResponse)
async def export_slides(req: ExportRequest):
    """Export a discovery session as slide-structured markdown."""
    runner = get_runner()

    try:
        state = await runner._workflow.aget_state(
            {"configurable": {"thread_id": req.session_id}}
        )
        if not state or not state.values:
            raise HTTPException(404, f"Session {req.session_id} not found")

        hypotheses = state.values.get("ranked_hypotheses", [])
        question = state.values.get("question", "")
        slides = _build_slides(question, hypotheses)

        return ExportResponse(
            session_id=req.session_id,
            format="slides",
            content=slides,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


def _build_report(question: str, hypotheses: list[dict]) -> str:
    """Build a markdown report from hypotheses."""
    parts = [
        f"# Scientific Discovery Report\n\n",
        f"## Question\n{question}\n\n",
    ]
    for i, h in enumerate(hypotheses, 1):
        parts.append(
            f"## Hypothesis {i}: {h.get('mechanism', 'Unknown')}\n\n"
            f"{h.get('description', '')}\n\n"
            f"- **Confidence:** {h.get('confidence', 0):.0%}\n"
            f"- **Diseases:** {', '.join(h.get('diseases', []))}\n"
            f"- **Genes:** {', '.join(h.get('genes', []))}\n"
            f"- **Next experiments:** {', '.join(h.get('next_experiments', []))}\n"
            f"- **Gaps:** {', '.join(h.get('gaps', []))}\n\n"
        )
    return "".join(parts)


def _build_slides(question: str, hypotheses: list[dict]) -> str:
    """Build slide-structured markdown for presentations."""
    slides = [
        f"# {question}\n\n---\n\n",
    ]
    for i, h in enumerate(hypotheses, 1):
        slides.append(
            f"## Slide {i}: {h.get('mechanism', 'Mechanism')}\n\n"
            f"{h.get('description', '')}\n\n"
            f"![Evidence](./figures/h{i}_evidence.png)\n\n"
            f"- Confidence: {h.get('confidence', 0):.0%}\n"
            f"- Diseases: {', '.join(h.get('diseases', []))}\n"
            f"- Gaps: {', '.join(h.get('gaps', []))}\n\n"
            f"---\n\n"
        )
    return "".join(slides)
