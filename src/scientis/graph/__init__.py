"""Neo4j graph driver management."""

import logging
from typing import Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

logger = logging.getLogger(__name__)

_driver: Optional[AsyncDriver] = None


def init_driver(uri: str, user: str, password: str) -> AsyncDriver:
    global _driver
    _driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    logger.info("Neo4j driver initialized: %s", uri)
    return _driver


def get_driver() -> AsyncDriver:
    if _driver is None:
        raise RuntimeError("Neo4j driver not initialized — call init_driver() first")
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver:
        await _driver.close()
        _driver = None
        logger.info("Neo4j driver closed")


async def verify_connection() -> bool:
    """Verify the Neo4j connection is alive."""
    try:
        driver = get_driver()
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS test")
            record = await result.single()
            return record is not None and record["test"] == 1
    except Exception:
        logger.exception("Neo4j connection verification failed")
        return False
