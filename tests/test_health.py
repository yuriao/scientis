"""Tests for the scientis API."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from scientis.api.deps import get_db as real_get_db
from scientis.main import create_app


@pytest.fixture
def app():
    app = create_app()

    # Override DB dependency so tests don't hit real PostgreSQL
    async def override_get_db():
        mock_session = MagicMock(spec=AsyncSession)
        mock_session.get = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock()
        yield mock_session

    app.dependency_overrides[real_get_db] = override_get_db
    return app


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_returns_ok(client):
    with patch("scientis.api.health.verify_connection", new_callable=AsyncMock, return_value=True):
        response = await client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert "dependencies" in data


@pytest.mark.asyncio
async def test_health_when_neo4j_down(client):
    """Health endpoint should still return 200 even when Neo4j is unavailable."""
    with patch("scientis.api.health.verify_connection", new_callable=AsyncMock, return_value=False):
        response = await client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["dependencies"]["neo4j"] == "unavailable"


@pytest.mark.asyncio
async def test_get_unknown_session_returns_404(client):
    response = await client.get("/v1/results/nonexistent-session")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_unknown_paper_returns_404(client):
    response = await client.get("/v1/papers/nonexistent-paper-id")
    assert response.status_code == 404
