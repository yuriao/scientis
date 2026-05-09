"""API router aggregation."""

from fastapi import APIRouter

from scientis.api.health import router as health_router
from scientis.api.papers import router as papers_router
from scientis.api.questions import router as questions_router
from scientis.api.exports import router as exports_router

router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(papers_router, tags=["papers"])
router.include_router(questions_router, tags=["discovery"])
router.include_router(exports_router, tags=["exports"])
