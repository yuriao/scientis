"""API router aggregation."""

from fastapi import APIRouter

from scientis.api.health import router as health_router
from scientis.api.papers import router as papers_router

router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(papers_router, tags=["papers"])
