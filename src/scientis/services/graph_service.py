"""Neo4j graph CRUD service.

Handles:
  - Creating Paper / Claim / Entity / Hypothesis nodes
  - Connecting claims to papers (PRESENTS_CLAIM)
  - Recording supporting and contradicting evidence
  - Cross-paper entity canonicalization
  - Subgraph retrieval for agent queries
"""

import logging
from typing import Optional

from neo4j import AsyncDriver

from scientis.graph.connection import get_driver
from scientis.models.claim import Claim
from scientis.models.hypothesis import Hypothesis
from scientis.models.paper import PaperSummary

logger = logging.getLogger(__name__)


class GraphService:
    """CRUD operations on the scientis evidence graph."""

    ENTITY_LABELS = {
        "Disease": "Disease",
        "Gene": "Gene",
        "Protein": "Protein",
        "Pathway": "Pathway",
        "Mechanism": "Mechanism",
        "Method": "Method",
        "Biomarker": "Biomarker",
        "CellType": "CellType",
        "Drug": "Drug",
        "Assay": "Assay",
    }

    def __init__(self, driver: Optional[AsyncDriver] = None):
        self._driver = driver

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            self._driver = get_driver()
        return self._driver

    # ── Paper ──────────────────────────────────────────────────────────────

    async def create_paper(self, paper: PaperSummary) -> None:
        async with self.driver.session() as session:
            await session.run(
                """
                MERGE (p:Paper {paper_id: $paper_id})
                SET p.title      = $title,
                    p.doi        = $doi,
                    p.year       = $year,
                    p.journal    = $journal,
                    p.status     = $status,
                    p.checksum   = $checksum,
                    p.updated_at = datetime()
                """,
                paper_id=paper.paper_id,
                title=paper.metadata.title,
                doi=paper.metadata.doi,
                year=paper.metadata.year,
                journal=paper.metadata.journal,
                status=paper.status,
                checksum=paper.checksum,
            )

    # ── Claims ───────────────────────────────────────────────────────────

    async def ingest_claims(self, paper_id: str, claims: list[Claim]) -> int:
        """Ingest all claims into the graph and return the count stored."""
        count = 0
        async with self.driver.session() as session:
            for claim in claims:
                # Merge the paper–claim relationship
                await session.run(
                    """
                    MATCH (p:Paper {paper_id: $paper_id})
                    MERGE (c:Claim {claim_id: $claim_id})
                    SET c.text               = $text,
                        c.section            = $section,
                        c.confidence         = $confidence,
                        c.updated_at         = datetime()
                    MERGE (p)-[:PRESENTS_CLAIM]->(c)
                    """,
                    paper_id=paper_id,
                    claim_id=claim.claim_id,
                    text=claim.claim,
                    section=claim.section,
                    confidence=claim.confidence,
                )

                # Link figure evidence
                for ev in claim.evidence:
                    if ev.figure_id:
                        await session.run(
                            """
                            MATCH (c:Claim {claim_id: $claim_id})
                            MERGE (f:Figure {figure_id: $figure_id})
                            SET f.paper_id = $paper_id, f.panel = $panel
                            MERGE (c)-[:SUPPORTED_BY_FIGURE {quote: $quote}]->(f)
                            """,
                            claim_id=claim.claim_id,
                            figure_id=ev.figure_id,
                            paper_id=paper_id,
                            panel=ev.panel,
                            quote=ev.quote,
                        )

                # Store contradicting text directly on the claim node
                # (a self-loop relationship is semantically meaningless)
                if claim.contradicting_evidence:
                    contradicting_quotes = " | ".join(
                        ce.quote for ce in claim.contradicting_evidence if ce.quote
                    )
                    if contradicting_quotes:
                        await session.run(
                            """
                            MATCH (c:Claim {claim_id: $claim_id})
                            SET c.contradicting_text = $text
                            """,
                            claim_id=claim.claim_id,
                            text=contradicting_quotes,
                        )

                count += 1

        logger.info("Ingested %d claims for paper %s", count, paper_id)
        return count

    # ── Entities ────────────────────────────────────────────────────────────

    async def create_entity(
        self,
        name: str,
        entity_type: str,
        aliases: Optional[list[str]] = None,
    ) -> None:
        """Create or update a canonical entity node."""
        label = self.ENTITY_LABELS.get(entity_type, "Entity")
        async with self.driver.session() as session:
            await session.run(
                f"""
                MERGE (e:{label} {{name: $name}})
                SET e.entity_type = $entity_type,
                    e.aliases     = $aliases,
                    e.updated_at  = datetime()
                """,
                name=name,
                entity_type=entity_type,
                aliases=aliases or [],
            )

    async def link_claim_entities(self, claim_id: str, entities: list[str]) -> None:
        """Link a claim to its mentioned entities via MENTIONS relationships."""
        async with self.driver.session() as session:
            for entity_name in entities:
                await session.run(
                    """
                    MATCH (c:Claim {claim_id: $claim_id})
                    MATCH (e) WHERE e.name = $entity_name
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    claim_id=claim_id,
                    entity_name=entity_name,
                )

    # ── Cross-paper queries ──────────────────────────────────────────────────

    async def find_supporting_claims(
        self, entity_name: str, limit: int = 20
    ) -> list[dict]:
        """Return claims that mention an entity, ordered by confidence."""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e {name: $name})<-[:MENTIONS]-(c:Claim)<-[:PRESENTS_CLAIM]-(p:Paper)
                RETURN c.claim_id  AS claim_id,
                       c.text      AS text,
                       c.confidence AS confidence,
                       p.paper_id  AS paper_id,
                       p.title     AS paper_title
                ORDER BY c.confidence DESC
                LIMIT $limit
                """,
                name=entity_name,
                limit=limit,
            )
            return [record.data() async for record in result]

    async def find_contradicting_claims(
        self, entity_name: str, limit: int = 20
    ) -> list[dict]:
        """Return claims about an entity that carry contradicting text."""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e {name: $name})<-[:MENTIONS]-(c:Claim)<-[:PRESENTS_CLAIM]-(p:Paper)
                WHERE c.contradicting_text IS NOT NULL AND c.contradicting_text <> ''
                RETURN c.claim_id          AS claim_id,
                       c.text              AS text,
                       c.confidence        AS confidence,
                       c.contradicting_text AS contradiction,
                       p.paper_id          AS paper_id
                LIMIT $limit
                """,
                name=entity_name,
                limit=limit,
            )
            return [record.data() async for record in result]

    # ── Subgraph retrieval ──────────────────────────────────────────────────

    async def get_mechanism_subgraph(
        self, disease_names: list[str], max_nodes: int = 100
    ) -> dict:
        """Retrieve claims linking diseases to other entities."""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (d:Disease)<-[:MENTIONS]-(c:Claim)<-[:PRESENTS_CLAIM]-(p:Paper)
                WHERE d.name IN $diseases
                RETURN d.name AS disease,
                       collect(DISTINCT {
                           claim_id: c.claim_id,
                           text:     c.text,
                           paper_id: p.paper_id
                       }) AS claims
                LIMIT $max_nodes
                """,
                diseases=disease_names,
                max_nodes=max_nodes,
            )
            records = [record.data() async for record in result]
            return {"diseases": disease_names, "subgraph": records}

    async def get_evidence_trail(self, claim_id: str) -> dict:
        """Return the full evidence chain for a claim (paper → claim → entities → figures)."""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (p:Paper)-[:PRESENTS_CLAIM]->(c:Claim {claim_id: $claim_id})
                OPTIONAL MATCH (c)-[:MENTIONS]->(e)
                OPTIONAL MATCH (c)-[:SUPPORTED_BY_FIGURE]->(f:Figure)
                RETURN p.paper_id  AS paper_id,
                       p.title     AS paper_title,
                       c.text      AS claim_text,
                       c.confidence AS confidence,
                       collect(DISTINCT e.name)      AS entities,
                       collect(DISTINCT f.figure_id) AS figures
                """,
                claim_id=claim_id,
            )
            record = await result.single()
            return record.data() if record else {}

    # ── Hypothesis ───────────────────────────────────────────────────────────

    async def create_hypothesis(self, hypothesis: Hypothesis) -> None:
        async with self.driver.session() as session:
            await session.run(
                """
                MERGE (h:Hypothesis {hypothesis_id: $hypothesis_id})
                SET h.mechanism   = $mechanism,
                    h.description = $description,
                    h.confidence  = $confidence,
                    h.status      = $status,
                    h.created_at  = datetime()
                """,
                hypothesis_id=hypothesis.hypothesis_id,
                mechanism=hypothesis.mechanism,
                description=hypothesis.description,
                confidence=hypothesis.confidence,
                status=hypothesis.status,
            )
            for cid in hypothesis.supporting_claims:
                await session.run(
                    """
                    MATCH (h:Hypothesis {hypothesis_id: $hid})
                    MATCH (c:Claim {claim_id: $cid})
                    MERGE (h)-[:SUPPORTED_BY]->(c)
                    """,
                    hid=hypothesis.hypothesis_id,
                    cid=cid,
                )
            for cid in hypothesis.contradicting_claims:
                await session.run(
                    """
                    MATCH (h:Hypothesis {hypothesis_id: $hid})
                    MATCH (c:Claim {claim_id: $cid})
                    MERGE (h)-[:CONTRADICTED_BY]->(c)
                    """,
                    hid=hypothesis.hypothesis_id,
                    cid=cid,
                )
            for disease in hypothesis.diseases:
                await session.run(
                    """
                    MATCH (h:Hypothesis {hypothesis_id: $hid})
                    MERGE (d:Disease {name: $name})
                    MERGE (h)-[:ASSOCIATED_WITH]->(d)
                    """,
                    hid=hypothesis.hypothesis_id,
                    name=disease,
                )


# ── Module-level singleton ───────────────────────────────────────────────

_graph_service: Optional[GraphService] = None


def get_graph_service() -> GraphService:
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
    return _graph_service
