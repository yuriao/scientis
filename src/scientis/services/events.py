"""Lightweight internal event bus.

In v0.1 this is an in-process pub/sub. Later: Redis pub/sub or Kafka.
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

Handler = Callable[[Any], None]
_handlers: dict[str, list[Handler]] = defaultdict(list)


class EventBus:
    """In-process typed event bus."""

    @staticmethod
    def subscribe(event_type: str, handler: Handler) -> None:
        _handlers[event_type].append(handler)

    @staticmethod
    def emit(event: Any) -> None:
        event_type = event.event_type
        logger.debug("Event emitted: %s id=%s", event_type, getattr(event, "paper_id", "?"))
        for handler in _handlers.get(event_type, []):
            try:
                handler(event)
            except Exception:
                logger.exception("Event handler failed: %s", event_type)


# Re-export event models for convenience
from scientis.models.events import ClaimExtracted, HypothesisGenerated, PaperParsed, PaperUploaded  # noqa: E402, F401
