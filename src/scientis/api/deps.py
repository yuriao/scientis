"""Shared API dependencies."""

from scientis.config import Settings, get_settings
from scientis.graph.connection import get_driver


def settings() -> Settings:
    return get_settings()
