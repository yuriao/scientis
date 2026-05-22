"""FastAPI application factory."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scientis.api import router as api_router
from scientis.config import get_settings
from scientis.db import init_db
from scientis.graph.connection import close_driver, init_driver


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise all stateful resources on startup and tear them down on shutdown."""
    settings = get_settings()

    # PostgreSQL — create tables if they don't exist yet
    await init_db()

    # Neo4j — open the driver and ensure graph constraints/indexes
    init_driver(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    from scientis.graph.schema import ensure_schema
    await ensure_schema()

    yield

    await close_driver()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Scientis",
        description="Agentic scientific discovery system",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=get_settings().api_prefix)
    return app


app = create_app()
