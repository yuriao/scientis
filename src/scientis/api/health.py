"""Health check endpoint."""

from fastapi import APIRouter

from scientis.graph.connection import verify_connection

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Return service liveness and dependency health."""
    neo4j_ok = False
    try:
        neo4j_ok = await verify_connection()
    except Exception:
        pass

    return {
        "status": "ok",
        "version": "0.1.0",
        "dependencies": {
            "neo4j": "ok" if neo4j_ok else "unavailable",
        },
    }
