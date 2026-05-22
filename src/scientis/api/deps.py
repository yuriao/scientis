"""Shared FastAPI dependencies."""

from scientis.config import Settings, get_settings
from scientis.db import get_db  # re-export so callers only need one import


def get_settings_dep() -> Settings:
    return get_settings()


__all__ = ["get_db", "get_settings_dep"]
