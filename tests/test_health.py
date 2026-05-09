"""Tests for scientis API."""

import pytest
from httpx import ASGITransport, AsyncClient

from scientis.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_list_papers_empty(client):
    response = await client.get("/v1/papers")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["items"] == []
