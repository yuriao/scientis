"""Async SQLAlchemy database setup.

Provides a module-level engine and session factory that are initialised
once from Settings and reused across the process lifetime.

Usage in FastAPI endpoints:
    async def my_route(db: AsyncSession = Depends(get_db)): ...
"""

import logging
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)

_engine = None
_session_factory: async_sessionmaker | None = None


class Base(DeclarativeBase):
    pass


def get_engine():
    global _engine
    if _engine is None:
        from scientis.config import get_settings

        _engine = create_async_engine(
            get_settings().database_url,
            echo=False,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields a database session per request."""
    async with get_session_factory()() as session:
        yield session


async def init_db() -> None:
    """Create all tables from the ORM metadata.

    For production, use Alembic migrations instead (see alembic.ini).
    This is called during application startup for development convenience.
    """
    from scientis.models import db_models  # ensure models are registered  # noqa: F401

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created (create_all)")
