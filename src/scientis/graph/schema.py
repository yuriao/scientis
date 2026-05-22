"""Neo4j schema management.

Creates constraints and indexes for the scientis evidence graph.
All entity node types use the `name` property as their unique key.
"""

import logging

from scientis.graph.connection import get_driver

logger = logging.getLogger(__name__)

SCHEMA_STATEMENTS = [
    # ── Unique constraints (node IDs) ──────────────────────────────────
    "CREATE CONSTRAINT paper_id      IF NOT EXISTS FOR (n:Paper)      REQUIRE n.paper_id      IS UNIQUE",
    "CREATE CONSTRAINT claim_id      IF NOT EXISTS FOR (n:Claim)      REQUIRE n.claim_id      IS UNIQUE",
    "CREATE CONSTRAINT hypothesis_id IF NOT EXISTS FOR (n:Hypothesis) REQUIRE n.hypothesis_id IS UNIQUE",
    "CREATE CONSTRAINT figure_id     IF NOT EXISTS FOR (n:Figure)     REQUIRE n.figure_id     IS UNIQUE",
    # Entity types — all keyed on `name`
    "CREATE CONSTRAINT disease_name   IF NOT EXISTS FOR (n:Disease)   REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT gene_name      IF NOT EXISTS FOR (n:Gene)      REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT protein_name   IF NOT EXISTS FOR (n:Protein)   REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT pathway_name   IF NOT EXISTS FOR (n:Pathway)   REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT mechanism_name IF NOT EXISTS FOR (n:Mechanism) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT method_name    IF NOT EXISTS FOR (n:Method)    REQUIRE n.name IS UNIQUE",
    # ── Indexes for common filter patterns ─────────────────────────────
    "CREATE INDEX paper_status      IF NOT EXISTS FOR (n:Paper)      ON (n.status)",
    "CREATE INDEX claim_confidence  IF NOT EXISTS FOR (n:Claim)      ON (n.confidence)",
    "CREATE INDEX hypothesis_status IF NOT EXISTS FOR (n:Hypothesis) ON (n.status)",
]


async def ensure_schema() -> None:
    """Apply graph schema (idempotent — skips existing constraints/indexes)."""
    driver = get_driver()
    async with driver.session() as session:
        for stmt in SCHEMA_STATEMENTS:
            try:
                await session.run(stmt)
            except Exception as exc:
                logger.debug("Schema statement skipped: %s — %s", stmt[:60], exc)
    logger.info("Graph schema ensured (%d statements)", len(SCHEMA_STATEMENTS))
